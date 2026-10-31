import Foundation

/// WebSocket client for the local Hushkey server (`/v1/stream`). See backend/hushkey/server.py.
final class BackendClient: NSObject, URLSessionWebSocketDelegate {
    enum Event {
        case connected([String: Any])
        case disconnected(String)
        case partial(String)
        case final([String: Any])
        case error(String)
    }

    var onEvent: ((Event) -> Void)?
    private(set) var isConnected = false

    private var url: URL
    private var session: URLSession!
    private var task: URLSessionWebSocketTask?
    private let sendQueue = DispatchQueue(label: "hushkey.ws.send")
    private var reconnectDelay: TimeInterval = 0.5
    private var wantConnection = false

    init(url: URL) {
        self.url = url
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func connect(to newURL: URL? = nil) {
        if let newURL { url = newURL }
        wantConnection = true
        task?.cancel(with: .goingAway, reason: nil)
        let t = session.webSocketTask(with: url)
        t.maximumMessageSize = 32 * 1024 * 1024
        task = t
        t.resume()
        receive(on: t)
    }

    func disconnect() {
        wantConnection = false
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        isConnected = false
    }

    func send(_ obj: [String: Any]) {
        sendQueue.async { [weak self] in
            guard let self, let task = self.task,
                  let data = try? JSONSerialization.data(withJSONObject: obj),
                  let text = String(data: data, encoding: .utf8) else { return }
            task.send(.string(text)) { error in
                if let error { self.onEvent?(.error("send failed: \(error.localizedDescription)")) }
            }
        }
    }

    private func receive(on t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            guard let self, t === self.task else { return }
            switch result {
            case .failure(let error):
                self.handleDrop(error.localizedDescription)
            case .success(let message):
                var data: Data?
                switch message {
                case .string(let s): data = s.data(using: .utf8)
                case .data(let d): data = d
                @unknown default: data = nil
                }
                if let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.dispatch(obj)
                }
                self.receive(on: t)
            }
        }
    }

    private func dispatch(_ obj: [String: Any]) {
        switch obj["type"] as? String {
        case "ready":
            isConnected = true
            reconnectDelay = 0.5
            onEvent?(.connected(obj["info"] as? [String: Any] ?? [:]))
        case "partial": onEvent?(.partial(obj["text"] as? String ?? ""))
        case "final": onEvent?(.final(obj))
        case "error": onEvent?(.error(obj["message"] as? String ?? "unknown error"))
        default: break
        }
    }

    private func handleDrop(_ reason: String) {
        let was = isConnected
        isConnected = false
        if was { onEvent?(.disconnected(reason)) }
        guard wantConnection else { return }
        let delay = reconnectDelay
        reconnectDelay = min(reconnectDelay * 2, 5)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.wantConnection else { return }
            self.connect()
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        if webSocketTask === task { handleDrop("closed (\(closeCode.rawValue))") }
    }

    /// One-shot GET /v1/health.
    static func health(_ url: URL) async -> [String: Any]? {
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let result = try? await URLSession.shared.data(for: req),
              (result.1 as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try? JSONSerialization.jsonObject(with: result.0) as? [String: Any]
    }
}
