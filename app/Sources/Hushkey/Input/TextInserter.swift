import AppKit
import ApplicationServices

/// Puts text at the cursor of the frontmost app.
enum TextInserter {
    /// Note: after rebuilding an ad-hoc-signed app, System Settings may still *show* Hushkey as
    /// allowed while macOS treats the new build as a different app and returns false here.
    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Shows the system prompt that sends the user to Privacy & Security ▸ Accessibility.
    @discardableResult
    static func requestTrust() -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    enum Outcome { case inserted, copiedOnly }

    /// Without Accessibility permission macOS silently drops synthetic keystrokes, so in that case
    /// we put the text on the clipboard and report `.copiedOnly` instead of failing invisibly.
    @discardableResult
    static func insert(_ text: String, mode: InsertMode) -> Outcome {
        guard !text.isEmpty else { return .inserted }
        guard isTrusted else {
            copy(text)
            return .copiedOnly
        }
        switch mode {
        case .paste: paste(text)
        case .type: type(text)
        }
        return .inserted
    }

    static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Clipboard + ⌘V, then restore the previous clipboard contents.
    private static func paste(_ text: String) {
        let pb = NSPasteboard.general
        let saved: [[NSPasteboard.PasteboardType: Data]] = (pb.pasteboardItems ?? []).map { item in
            var d: [NSPasteboard.PasteboardType: Data] = [:]
            for t in item.types { if let data = item.data(forType: t) { d[t] = data } }
            return d
        }
        pb.clearContents()
        pb.setString(text, forType: .string)
        let changeCount = pb.changeCount

        // Full ⌘ down → V down → V up → ⌘ up sequence: some apps (Electron, Java) check the
        // modifier key events rather than only the flags on the V event.
        let src = CGEventSource(stateID: .hidSystemState)
        let cmdKey: CGKeyCode = 0x37, vKey: CGKeyCode = 0x09
        let events: [(CGKeyCode, Bool, CGEventFlags)] = [
            (cmdKey, true, .maskCommand), (vKey, true, .maskCommand), (vKey, false, .maskCommand), (cmdKey, false, []),
        ]
        for (key, down, flags) in events {
            let e = CGEvent(keyboardEventSource: src, virtualKey: key, keyDown: down)
            e?.flags = flags
            e?.post(tap: .cghidEventTap)
            usleep(8000)
        }

        // Slow apps read the clipboard late; wait before restoring the user's clipboard.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            guard pb.changeCount == changeCount else { return }  // user copied something meanwhile
            pb.clearContents()
            let items: [NSPasteboardItem] = saved.map { d in
                let item = NSPasteboardItem()
                for (t, data) in d { item.setData(data, forType: t) }
                return item
            }
            if !items.isEmpty { pb.writeObjects(items) }
        }
    }

    /// Synthesised Unicode keystrokes (works where paste is blocked, slower for long text).
    private static func type(_ text: String) {
        let src = CGEventSource(stateID: .hidSystemState)
        let utf16 = Array(text.utf16)
        var i = 0
        while i < utf16.count {
            let chunk = Array(utf16[i..<min(i + 16, utf16.count)])
            let down = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: true)
            let up = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: false)
            chunk.withUnsafeBufferPointer { buf in
                down?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
                up?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
            }
            down?.post(tap: .cghidEventTap)
            up?.post(tap: .cghidEventTap)
            i += 16
            usleep(2000)
        }
    }
}
