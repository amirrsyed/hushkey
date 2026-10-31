import Foundation

/// UserDefaults keys and defaults. Views bind with `@AppStorage(AppSettings.Key.x)`.
enum AppSettings {
    enum Key {
        static let serverURL = "serverURL"
        static let autoLaunchBackend = "autoLaunchBackend"
        static let backendCommand = "backendCommand"
        static let hotkey = "hotkey"
        static let insertMode = "insertMode"
        static let trailingSpace = "trailingSpace"
        static let handsFree = "handsFree"
        static let showPartials = "showPartials"
        static let cameraID = "cameraID"
        static let showHUDPreview = "showHUDPreview"
        static let onboarded = "onboarded"
    }

    /// ~/Library/Application Support/Hushkey — models, adapters, enrollment clips and the server's venv.
    static var hushkeyHome: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Hushkey")
    }

    /// Where `install_backend.sh` puts the model server.
    static var venvPython: URL { hushkeyHome.appendingPathComponent("venv/bin/python") }
    static var backendInstalled: Bool { FileManager.default.isExecutableFile(atPath: venvPython.path) }

    static var defaultBackendCommand: String {
        backendInstalled ? "\"\(venvPython.path)\" -m hushkey.cli serve" : "hushkey serve"
    }

    /// The model-server installer bundled inside Hushkey.app (see scripts/build_app.sh).
    static var bundledInstaller: URL? {
        guard let url = Bundle.main.resourceURL?.appendingPathComponent("install_backend.sh"),
              FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    static func registerDefaults() {
        // A saved command that `cd`s into a folder that no longer exists (e.g. a deleted download)
        // is replaced by the standard install location when that exists.
        if backendInstalled, let saved = UserDefaults.standard.string(forKey: Key.backendCommand),
           let r = saved.range(of: "cd \""), let end = saved[r.upperBound...].firstIndex(of: "\"") {
            let dir = String(saved[r.upperBound..<end])
            if !FileManager.default.fileExists(atPath: dir) {
                UserDefaults.standard.removeObject(forKey: Key.backendCommand)
            }
        }
        UserDefaults.standard.register(defaults: [
            Key.serverURL: "ws://127.0.0.1:8765/v1/stream",
            Key.autoLaunchBackend: true,
            Key.backendCommand: defaultBackendCommand,
            Key.hotkey: Hotkey.rightOption.rawValue,
            Key.insertMode: InsertMode.paste.rawValue,
            Key.trailingSpace: true,
            Key.handsFree: false,
            Key.showPartials: true,
            Key.cameraID: "",
            Key.showHUDPreview: true,
            Key.onboarded: false,
        ])
    }

    static var serverURL: URL {
        URL(string: UserDefaults.standard.string(forKey: Key.serverURL) ?? "") ?? URL(string: "ws://127.0.0.1:8765/v1/stream")!
    }

    /// http(s) base for the REST endpoints, derived from the WebSocket URL.
    static var healthURL: URL {
        var c = URLComponents(url: serverURL, resolvingAgainstBaseURL: false)!
        c.scheme = (c.scheme == "wss") ? "https" : "http"
        c.path = "/v1/health"
        return c.url!
    }

    static var hotkey: Hotkey { Hotkey(rawValue: UserDefaults.standard.string(forKey: Key.hotkey) ?? "") ?? .rightOption }
    static var insertMode: InsertMode { InsertMode(rawValue: UserDefaults.standard.string(forKey: Key.insertMode) ?? "") ?? .paste }
    static var trailingSpace: Bool { UserDefaults.standard.bool(forKey: Key.trailingSpace) }
    static var handsFree: Bool { UserDefaults.standard.bool(forKey: Key.handsFree) }
    static var showPartials: Bool { UserDefaults.standard.bool(forKey: Key.showPartials) }
    static var autoLaunchBackend: Bool { UserDefaults.standard.bool(forKey: Key.autoLaunchBackend) }
    static var backendCommand: String { UserDefaults.standard.string(forKey: Key.backendCommand) ?? defaultBackendCommand }
    static var cameraID: String { UserDefaults.standard.string(forKey: Key.cameraID) ?? "" }
}

enum Hotkey: String, CaseIterable, Identifiable {
    case rightOption, rightCommand, fn
    var id: String { rawValue }
    var label: String {
        switch self {
        case .rightOption: return "Right ⌥ Option"
        case .rightCommand: return "Right ⌘ Command"
        case .fn: return "fn / 🌐"
        }
    }
}

enum InsertMode: String, CaseIterable, Identifiable {
    case paste, type
    var id: String { rawValue }
    var label: String { self == .paste ? "Paste (fast, restores clipboard)" : "Type keystrokes" }
}
