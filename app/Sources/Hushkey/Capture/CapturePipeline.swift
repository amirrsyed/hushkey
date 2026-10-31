import CoreGraphics
import CoreVideo
import Foundation

/// Everything that runs on the camera queue. Owns the camera, the per-frame processor, the
/// hands-free motion gate and a short pre-roll so the first syllable isn't lost.
final class CapturePipeline {
    struct Update {
        let faceVisible: Bool
        let preview: CGImage?
        let motion: Double
        let gate: MotionGate.Event
    }

    private struct State {
        var recording = false
        var handsFree = false
        var wantPreview = true
        var preroll: [(t: Double, msg: [String: Any])] = []
    }

    let camera = CameraService()
    private let processor = FrameProcessor()
    private let gate = MotionGate()
    private let client: BackendClient
    private let state = Locked(State())
    var onUpdate: ((Update) -> Void)?

    static let prerollSeconds = 0.4

    init(client: BackendClient) {
        self.client = client
        camera.onFrame = { [weak self] pb, t in self?.handle(pb, t) }
    }

    var isCameraRunning: Bool { camera.isRunning }

    func startCamera(cameraID: String) throws {
        if !camera.isRunning { try camera.start(cameraID: cameraID) }
    }

    func stopCamera() {
        if camera.isRunning { camera.stop() }
        state.mutate { $0.preroll.removeAll() }
    }

    func setHandsFree(_ on: Bool) {
        state.mutate { $0.handsFree = on }
        camera.queue.async { [gate] in gate.calibrate(); gate.reset() }
    }

    func setPreviewWanted(_ on: Bool) { state.mutate { $0.wantPreview = on } }

    /// Sends `start`, flushes the pre-roll and switches to live streaming — atomically with
    /// respect to the camera queue so no frame is lost or duplicated.
    func beginUtterance(_ start: [String: Any]) {
        state.mutate { s in
            client.send(start)
            for f in s.preroll { client.send(f.msg) }
            s.preroll.removeAll()
            s.recording = true
        }
    }

    func endUtterance(_ message: [String: Any]) {
        state.mutate { s in
            s.recording = false
            client.send(message)
        }
        camera.queue.async { [gate] in gate.reset() }
    }

    private func handle(_ pb: CVPixelBuffer, _ t: Double) {
        let wantPreview = state.get().wantPreview
        let out = processor.process(pb, time: t, encode: true, wantPreview: wantPreview)
        var handsFree = false
        if let wire = out.wire {
            state.mutate { s in
                handsFree = s.handsFree
                if s.recording {
                    client.send(wire)
                } else {
                    s.preroll.append((t, wire))
                    while let first = s.preroll.first, t - first.t > Self.prerollSeconds { s.preroll.removeFirst() }
                }
            }
        }
        // Always run the gate so the HUD motion meter works in push-to-talk too; only act on its
        // segment events in hands-free mode.
        let gateEvent = gate.ingest(t: t, articulation: out.keypoints?.articulation)
        let event: MotionGate.Event = handsFree ? gateEvent : .none
        let level = min(1, gate.energy / max((gate.noiseFloor ?? 0.002) * gate.triggerRatio, 1e-4))
        onUpdate?(Update(faceVisible: out.keypoints != nil, preview: out.mouthPreview, motion: level, gate: event))
    }
}
