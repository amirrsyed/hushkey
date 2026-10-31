import AppKit

/// Wispr-style push-to-talk on a modifier key:
///   • hold the key → record while held
///   • double-tap   → hands-free lock (records until the next tap)
///   • Esc while recording → cancel
/// Uses NSEvent global + local monitors (needs Accessibility permission for global events).
final class HotkeyMonitor {
    var onPress: (() -> Void)?
    var onRelease: (() -> Void)?
    var onCancel: (() -> Void)?
    var onLockToggle: ((Bool) -> Void)?

    var hotkey: Hotkey = .rightOption
    private(set) var locked = false

    private var monitors: [Any] = []
    private var isDown = false
    private var lastTap: Date = .distantPast
    private var downAt: Date = .distantPast
    private static let doubleTapWindow: TimeInterval = 0.35
    private static let tapMaxDuration: TimeInterval = 0.25

    func start() {
        stop()
        if let g = NSEvent.addGlobalMonitorForEvents(matching: [.flagsChanged, .keyDown], handler: { [weak self] e in self?.handle(e) }) {
            monitors.append(g)
        }
        if let l = NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged, .keyDown], handler: { [weak self] e in self?.handle(e); return e }) {
            monitors.append(l)
        }
    }

    func stop() {
        monitors.forEach { NSEvent.removeMonitor($0) }
        monitors.removeAll()
    }

    private var keyCode: UInt16 {
        switch hotkey {
        case .rightOption: return 61
        case .rightCommand: return 54
        case .fn: return 63
        }
    }

    private func isPressed(_ e: NSEvent) -> Bool {
        switch hotkey {
        case .rightOption: return e.modifierFlags.contains(.option)
        case .rightCommand: return e.modifierFlags.contains(.command)
        case .fn: return e.modifierFlags.contains(.function)
        }
    }

    private func handle(_ e: NSEvent) {
        if e.type == .keyDown {
            if e.keyCode == 53, isDown || locked {  // Esc
                isDown = false
                locked = false
                onCancel?()
            } else if isDown {
                // A real shortcut (e.g. ⌥+letter) – not dictation. Abort quietly.
                isDown = false
                onCancel?()
            }
            return
        }
        guard e.type == .flagsChanged, e.keyCode == keyCode else { return }
        let pressed = isPressed(e)
        let now = Date()
        if pressed && !isDown {
            isDown = true
            downAt = now
            if locked {  // tap while locked → finish
                locked = false
                onLockToggle?(false)
                onRelease?()
                isDown = false
                return
            }
            onPress?()
        } else if !pressed && isDown {
            isDown = false
            let wasTap = now.timeIntervalSince(downAt) < Self.tapMaxDuration
            if wasTap && now.timeIntervalSince(lastTap) < Self.doubleTapWindow {
                locked = true  // second quick tap: keep recording hands-free
                onLockToggle?(true)
                lastTap = .distantPast
                return
            }
            if wasTap {
                lastTap = now
                onCancel?()  // a single short tap is too short to be speech
                return
            }
            onRelease?()
        }
    }
}
