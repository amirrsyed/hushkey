import SwiftUI

struct OnboardingView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismissWindow) private var dismissWindow
    @AppStorage(Settings.Key.onboarded) private var onboarded = false

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
                detail: "Needed to notice the push-to-talk key in any app and to type the result at your cursor.",
                action: "Open Accessibility settings"
            ) { model.requestAccessibility() }
            step(
                done: model.backendConnected,
                title: "Local model server",
                detail: "Install the backend once: `cd backend && pip install -e . && hushkey download`. Hushkey starts it for you.",
                action: "Retry connection"
            ) { Task { await model.connectBackend() } }
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
        .frame(width: 560, height: 440)
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
