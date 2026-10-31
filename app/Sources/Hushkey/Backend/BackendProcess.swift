import Foundation

/// Optionally launches and supervises the Python backend (`hushkey serve`) and runs one-off
/// commands such as `hushkey train personalize`. Commands run through a login shell so that
/// the user's PATH (venv, conda, uv, pyenv) applies.
final class BackendProcess {
    private var server: Process?
    private(set) var log = Locked<[String]>([])
    var onLog: ((String) -> Void)?

    var isRunning: Bool { server?.isRunning ?? false }

    func startServer(command: String) {
        guard !isRunning else { return }
        server = run(command, onLine: { [weak self] line in self?.append(line) }, onExit: { [weak self] code in
            self?.append("backend exited with status \(code)")
        })
    }

    func stopServer() {
        server?.terminate()
        server = nil
    }

    func restartServer(command: String) {
        stopServer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in self?.startServer(command: command) }
    }

    @discardableResult
    func run(_ command: String, onLine: @escaping (String) -> Void, onExit: @escaping (Int32) -> Void) -> Process? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", command]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        let buffer = Locked(Data())
        pipe.fileHandleForReading.readabilityHandler = { h in
            let chunk = h.availableData
            guard !chunk.isEmpty else { return }
            let lines: [String] = buffer.mutate { buf in
                buf.append(chunk)
                var out: [String] = []
                while let nl = buf.firstIndex(of: 0x0A) {
                    if let line = String(data: buf.subdata(in: buf.startIndex..<nl), encoding: .utf8) { out.append(line) }
                    buf.removeSubrange(buf.startIndex...nl)
                }
                return out
            }
            if !lines.isEmpty { DispatchQueue.main.async { lines.forEach(onLine) } }
        }
        p.terminationHandler = { proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async { onExit(proc.terminationStatus) }
        }
        do {
            try p.run()
            return p
        } catch {
            onLine("failed to launch `\(command)`: \(error.localizedDescription)")
            return nil
        }
    }

    private func append(_ line: String) {
        log.mutate { l in
            l.append(line)
            if l.count > 500 { l.removeFirst(l.count - 500) }
        }
        onLog?(line)
    }
}
