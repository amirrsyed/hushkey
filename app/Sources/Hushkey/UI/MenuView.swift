import SwiftUI

struct MenuView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            if !model.cameraAuthorized || !model.accessibilityTrusted {
                Button {
                    open("onboarding")
                } label: {
                    Label("Finish setup: grant permissions", systemImage: "exclamationmark.shield")
                }
                .buttonStyle(.borderedProminent)
            }
            Text("Hold **\(Settings.hotkey.label)** and silently mouth. Release to type. Double-tap to go hands-free, Esc to cancel.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Toggle(isOn: Binding(get: { model.autoMode }, set: { model.setAutoMode($0) })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Auto-detect mouthing")
                    Text("Starts and stops on lip/jaw motion. The camera stays on.").font(.caption).foregroundStyle(.secondary)
                }
            }

            if !model.history.isEmpty {
                Divider()
                Text("Recent").font(.caption).foregroundStyle(.secondary)
                ForEach(model.history.prefix(4)) { d in
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(d.text, forType: .string)
                    } label: {
                        HStack {
                            Text(d.text).lineLimit(1).truncationMode(.tail)
                            Spacer()
                            Text("\(d.latencyMs) ms").font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                    .buttonStyle(.plain)
                    .help("Click to copy · raw: \(d.raw)")
                }
            }
            Divider()
            HStack {
                Button("Train on my mouth…") { open("enroll") }
                Spacer()
                SettingsLink { Text("Settings…") }
            }
            HStack {
                Button("Restart backend") { model.restartBackend() }
                Spacer()
                Button("Quit") { model.shutdown(); NSApp.terminate(nil) }
            }
        }
        .padding(14)
        .frame(width: 340)
        .onAppear { model.refreshPermissions() }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "mouth").font(.system(size: 22, weight: .medium))
            VStack(alignment: .leading, spacing: 2) {
                Text("Hushkey").font(.headline)
                HStack(spacing: 6) {
                    Circle().fill(model.backendConnected ? Color.green : Color.orange).frame(width: 7, height: 7)
                    Text(backendStatus).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            Spacer()
        }
    }

    private var backendStatus: String {
        guard model.backendConnected else { return "Backend starting / not connected" }
        let dev = model.backendInfo["device"] as? String ?? "?"
        let adapter = (model.backendInfo["adapter"] as? String).map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent } ?? "base model"
        let tiny = (model.backendInfo["tiny"] as? Bool) == true ? " · TEST MODEL" : ""
        return "\(dev) · \(adapter)\(tiny)"
    }

    private func open(_ id: String) {
        openWindow(id: id)
        NSApp.activate(ignoringOtherApps: true)
    }
}
