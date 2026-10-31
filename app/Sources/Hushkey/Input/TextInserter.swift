import AppKit
import ApplicationServices

/// Puts text at the cursor of the frontmost app.
enum TextInserter {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Shows the system prompt that sends the user to Privacy & Security ▸ Accessibility.
    @discardableResult
    static func requestTrust() -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    static func insert(_ text: String, mode: InsertMode) {
        guard !text.isEmpty else { return }
        switch mode {
        case .paste: paste(text)
        case .type: type(text)
        }
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

        let src = CGEventSource(stateID: .combinedSessionState)
        let vKey: CGKeyCode = 9  // kVK_ANSI_V
        let down = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
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
        let src = CGEventSource(stateID: .combinedSessionState)
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
