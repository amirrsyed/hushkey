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

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            Key.serverURL: "ws://127.0.0.1:8765/v1/stream",
            Key.autoLaunchBackend: true,
            Key.backendCommand: "hushkey serve",
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
    static var backendCommand: String { UserDefaults.standard.string(forKey: Key.backendCommand) ?? "hushkey serve" }
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
