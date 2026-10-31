import AppKit
import Observation
import SwiftUI

struct Dictation: Identifiable {
    let id = UUID()
    let date: Date
    let text: String
    let raw: String
    let latencyMs: Int
    let app: String?
}

@MainActor @Observable
final class AppModel {
    enum Phase: Equatable {
        case idle, listening, transcribing
        case error(String)
    }

    // UI state
    var phase: Phase = .idle
    var partial = ""
    var lastText = ""
    var history: [Dictation] = []
    var backendConnected = false
    var backendInfo: [String: Any] = [:]
    var backendLog: [String] = []
    var faceVisible = false
    var mouthPreview: CGImage?
    var motionLevel = 0.0
    var cameraAuthorized = CameraService.authorization == .authorized
    var accessibilityTrusted = TextInserter.isTrusted
    var handsFreeLocked = false
    var autoMode = Settings.handsFree

    // Enrollment ("Train on my mouth")
    var enrollmentPrompt: String?  // non-nil while the enrollment window is open
    var lastEnrollment: (prompt: String, heard: String)?
    var enrolledThisSession = 0
    var trainingLog: [String] = []
    var isTraining = false

    @ObservationIgnored let client = BackendClient(url: Settings.serverURL)
    @ObservationIgnored lazy var pipeline = CapturePipeline(client: client)
    @ObservationIgnored let backend = BackendProcess()
    @ObservationIgnored let hotkeys = HotkeyMonitor()
    @ObservationIgnored lazy var hud = HUDController(model: self)
    @ObservationIgnored private var idleCameraTask: Task<Void, Never>?
    @ObservationIgnored private var finalTimeoutTask: Task<Void, Never>?
    @ObservationIgnored private var sessionApp: String?
    @ObservationIgnored private var sessionEnrollText: String?
    @ObservationIgnored private var started = false

    static let cameraIdleSeconds: TimeInterval = 30

    func start() {
        guard !started else { return }
        started = true
        Settings.registerDefaults()
        // These fire on background queues (URLSession, camera); hop to the main actor.
        client.onEvent = { @Sendable [weak self] ev in Task { @MainActor in self?.handle(ev) } }
        pipeline.onUpdate = { @Sendable [weak self] u in Task { @MainActor in self?.apply(u) } }
        backend.onLog = { @Sendable [weak self] line in Task { @MainActor in self?.appendLog(line) } }

        hotkeys.hotkey = Settings.hotkey
        hotkeys.onPress = { [weak self] in self?.beginDictation() }
        hotkeys.onRelease = { [weak self] in self?.endDictation() }
        hotkeys.onCancel = { [weak self] in self?.cancelDictation() }
        hotkeys.onLockToggle = { [weak self] locked in self?.handsFreeLocked = locked }
        hotkeys.start()

        Task { await connectBackend() }
        if autoMode { setAutoMode(true) }
    }

    func shutdown() {
        hotkeys.stop()
        pipeline.stopCamera()
        client.disconnect()
        if Settings.autoLaunchBackend { backend.stopServer() }
    }

    // MARK: Backend

    func connectBackend() async {
        if await BackendClient.health(Settings.healthURL) == nil, Settings.autoLaunchBackend, !backend.isRunning {
            appendLog("starting backend: \(Settings.backendCommand)")
            backend.startServer(command: Settings.backendCommand)
            // The model takes a few seconds to load; the client keeps retrying with backoff.
        }
        client.connect(to: Settings.serverURL)
    }

    func restartBackend() {
        client.disconnect()
        backendConnected = false
        backend.restartServer(command: Settings.backendCommand)
        after(2) { [weak self] in self?.client.connect(to: Settings.serverURL) }
    }

    private func appendLog(_ line: String) {
        backendLog.append(line)
        if backendLog.count > 300 { backendLog.removeFirst(backendLog.count - 300) }
    }

    // MARK: Permissions

    func refreshPermissions() {
        cameraAuthorized = CameraService.authorization == .authorized
        accessibilityTrusted = TextInserter.isTrusted
    }

    func requestCamera() async {
        cameraAuthorized = await CameraService.requestAccess()
    }

    func requestAccessibility() {
        TextInserter.requestTrust()
        // The grant happens in System Settings; poll for a while.
        Task {
            for _ in 0..<60 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if TextInserter.isTrusted { accessibilityTrusted = true; hotkeys.start(); return }
            }
        }
    }

    // MARK: Dictation lifecycle

    /// Run `body` on the main actor after `seconds`; returns a cancellable task.
    @discardableResult
    private func after(_ seconds: Double, _ body: @escaping @MainActor () -> Void) -> Task<Void, Never> {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            if !Task.isCancelled { body() }
        }
    }

    private func ensureCamera() -> Bool {
        idleCameraTask?.cancel()
        guard cameraAuthorized else {
            fail("Camera permission needed — open Hushkey ▸ Permissions.")
            return false
        }
        do {
            try pipeline.startCamera(cameraID: Settings.cameraID)
            return true
        } catch {
            fail(error.localizedDescription)
            return false
        }
    }

    private func scheduleCameraIdle() {
        idleCameraTask?.cancel()
        guard !autoMode, enrollmentPrompt == nil else { return }
        idleCameraTask = after(Self.cameraIdleSeconds) { [weak self] in
            guard let self, !self.isBusy, !self.autoMode, self.enrollmentPrompt == nil else { return }
            self.pipeline.stopCamera()
        }
    }

    var isBusy: Bool { phase == .listening || phase == .transcribing }

    func beginDictation() {
        guard !isBusy else { return }  // one utterance at a time
        guard backendConnected else {
            fail("Backend not connected — is `hushkey serve` running?")
            return
        }
        guard ensureCamera() else { return }
        sessionApp = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        sessionEnrollText = enrollmentPrompt
        partial = ""
        phase = .listening
        hud.show()
        var start: [String: Any] = ["type": "start", "partials": Settings.showPartials]
        if let app = sessionApp { start["app"] = app }
        if let e = sessionEnrollText { start["enroll_text"] = e }
        pipeline.beginUtterance(start)
    }

    func endDictation() {
        guard phase == .listening else { return }
        phase = .transcribing
        handsFreeLocked = false
        pipeline.endUtterance(["type": "stop"])
        finalTimeoutTask?.cancel()
        finalTimeoutTask = after(30) { [weak self] in
            if self?.phase == .transcribing { self?.fail("Timed out waiting for the backend.") }
        }
    }

    func cancelDictation() {
        guard isBusy else { return }
        pipeline.endUtterance(["type": "cancel"])
        handsFreeLocked = false
        phase = .idle
        partial = ""
        hud.hide(after: 0)
        scheduleCameraIdle()
    }

    func setAutoMode(_ on: Bool) {
        autoMode = on
        UserDefaults.standard.set(on, forKey: Settings.Key.handsFree)
        pipeline.setHandsFree(on)
        if on {
            _ = ensureCamera()
        } else {
            scheduleCameraIdle()
        }
    }

    private func fail(_ message: String) {
        phase = .error(message)
        hud.show()
        hud.hide(after: 3)
        after(3) { [weak self] in
            if case .error = self?.phase { self?.phase = .idle }
        }
    }

    // MARK: Events

    private func apply(_ u: CapturePipeline.Update) {
        faceVisible = u.faceVisible
        if let p = u.preview { mouthPreview = p }
        motionLevel = u.motion
        guard autoMode, !hotkeys.locked else { return }
        switch u.gate {
        case .open: if phase == .idle { beginDictation() }
        case .close: if phase == .listening { endDictation() }
        case .discard: if phase == .listening { cancelDictation() }
        case .none: break
        }
    }

    private func handle(_ ev: BackendClient.Event) {
        switch ev {
        case .connected(let info):
            backendConnected = true
            backendInfo = info
        case .disconnected(let reason):
            backendConnected = false
            appendLog("disconnected: \(reason)")
            if isBusy { fail("Backend disconnected.") }
        case .partial(let text):
            if phase == .listening { partial = text }
        case .final(let obj):
            finalTimeoutTask?.cancel()
            let text = (obj["text"] as? String) ?? ""
            let raw = (obj["raw"] as? String) ?? ""
            let latency = (obj["latency_ms"] as? Int) ?? 0
            if let prompt = sessionEnrollText {
                lastEnrollment = (prompt, raw)
                if obj["enrolled"] != nil { enrolledThisSession += 1 }
            } else if !text.isEmpty {
                let out = Settings.trailingSpace ? text + " " : text
                TextInserter.insert(out, mode: Settings.insertMode)
                history.insert(Dictation(date: Date(), text: text, raw: raw, latencyMs: latency, app: sessionApp), at: 0)
                if history.count > 50 { history.removeLast() }
            }
            lastText = text.isEmpty ? "(nothing recognised)" : text
            partial = ""
            phase = .idle
            hud.hide(after: 1.2)
            scheduleCameraIdle()
        case .error(let message):
            finalTimeoutTask?.cancel()
            fail(message)
            scheduleCameraIdle()
        }
    }

    // MARK: Enrollment + training

    func openEnrollment(prompt: String) {
        enrollmentPrompt = prompt
        _ = ensureCamera()
    }

    func closeEnrollment() {
        enrollmentPrompt = nil
        scheduleCameraIdle()
    }

    func trainPersonalAdapter() {
        guard !isTraining else { return }
        isTraining = true
        trainingLog = ["$ hushkey train personalize"]
        // Reuse whatever environment prefix the serve command has (cd …, uv run, conda run …).
        let serve = Settings.backendCommand
        let prefix = serve.range(of: "hushkey serve").map { String(serve[..<$0.lowerBound]) } ?? ""
        let cmd = prefix + "hushkey train personalize"
        backend.run(cmd, onLine: { [weak self] line in self?.trainingLog.append(line) }, onExit: { [weak self] code in
            guard let self else { return }
            self.isTraining = false
            self.trainingLog.append(code == 0 ? "✓ done — restarting backend to load the new adapter" : "✗ training failed (exit \(code))")
            if code == 0 { self.restartBackend() }
        })
    }
}
