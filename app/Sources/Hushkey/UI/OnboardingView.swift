import SwiftUI

struct OnboardingView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismissWindow) private var dismissWindow
    @AppStorage(AppSettings.Key.onboarded) private var onboarded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                Image(systemName: "mouth").font(.system(size: 34))
                VStack(alignment: .leading) {
                    Text("Welcome to Hushkey").font(.title2.bold())
                    Text("Dictation from silent lip, jaw and throat movement. No microphone, nothing leaves your Mac.")
                        .foregroundStyle(.secondary)
                }
            }
            step(
                done: model.cameraAuthorized,
                title: "Camera",
                detail: "Hushkey reads your mouth from the webcam. Frames are processed locally and never stored, except clips you record in “Train on my mouth”.",
                action: "Allow camera"
            ) { Task { await model.requestCamera() } }
            step(
                done: model.accessibilityTrusted,
                title: "Accessibility",
                detail: "Needed to notice the push-to-talk key in any app and to type the result at your cursor. If Hushkey is already listed but this isn't ticked, remove it with “–” and add it again: macOS treats each new build as a new app.",
                action: "Open Accessibility settings"
            ) { model.requestAccessibility() }
            if model.backendInstalled || AppSettings.bundledInstaller == nil {
                step(
                    done: model.backendConnected,
                    title: "Local model server",
                    detail: model.backendInstalled
                        ? "Installed. It takes 10–30 s to load the model after Hushkey starts it."
                        : "Not installed yet. From the source folder run `./install.sh` in Terminal, then click Retry.",
                    action: "Retry connection"
                ) { Task { await model.connectBackend() } }
            } else {
                step(
                    done: false,
                    title: "Local model server",
                    detail: "One-time download of the lip-reading model and its runtime (≈1.5 GB, a few minutes). Everything stays on this Mac.",
                    action: model.isInstallingBackend ? "Installing…" : "Install model server"
                ) { model.installBackend() }
            }
            if model.isInstallingBackend || !model.installLog.isEmpty {
                HStack(spacing: 8) {
                    if model.isInstallingBackend { ProgressView().controlSize(.small) }
                    Text(model.installLog.last ?? "")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
                .padding(.leading, 32)
            }
            Spacer()
            HStack {
                Text("Tip: sit facing the camera with your mouth and neck visible and even lighting.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Done") { onboarded = true; dismissWindow(id: "onboarding") }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 580, height: 500)
        .onAppear { model.refreshPermissions() }
    }

    @ViewBuilder
    private func step(done: Bool, title: String, detail: String, action: String, perform: @escaping () -> Void) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(done ? Color.green : Color.secondary)
                .font(.title3)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if !done { Button(action, action: perform).padding(.top, 2) }
            }
        }
    }
}
