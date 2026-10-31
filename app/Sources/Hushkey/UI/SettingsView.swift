import AVFoundation
import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @AppStorage(Settings.Key.serverURL) private var serverURL = "ws://127.0.0.1:8765/v1/stream"
    @AppStorage(Settings.Key.autoLaunchBackend) private var autoLaunch = true
    @AppStorage(Settings.Key.backendCommand) private var backendCommand = "hushkey serve"
    @AppStorage(Settings.Key.hotkey) private var hotkey = Hotkey.rightOption.rawValue
    @AppStorage(Settings.Key.insertMode) private var insertMode = InsertMode.paste.rawValue
    @AppStorage(Settings.Key.trailingSpace) private var trailingSpace = true
    @AppStorage(Settings.Key.showPartials) private var showPartials = true
    @AppStorage(Settings.Key.showHUDPreview) private var showPreview = true
    @AppStorage(Settings.Key.cameraID) private var cameraID = ""

    var body: some View {
        TabView {
            Form {
                Picker("Push-to-talk key", selection: $hotkey) {
                    ForEach(Hotkey.allCases) { Text($0.label).tag($0.rawValue) }
                }
                .onChange(of: hotkey) { _, new in model.hotkeys.hotkey = Hotkey(rawValue: new) ?? .rightOption }
                Picker("Insert text by", selection: $insertMode) {
                    ForEach(InsertMode.allCases) { Text($0.label).tag($0.rawValue) }
                }
                Toggle("Add a space after each dictation", isOn: $trailingSpace)
                Toggle("Show live partial transcript", isOn: $showPartials)
                Toggle("Show mouth preview in the HUD", isOn: $showPreview)
                Picker("Camera", selection: $cameraID) {
                    Text("System default").tag("")
                    ForEach(CameraService.availableCameras(), id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
                }
                .onChange(of: cameraID) { _, _ in model.pipeline.stopCamera() }
            }
            .tabItem { Label("General", systemImage: "gearshape") }

            Form {
                TextField("Server WebSocket URL", text: $serverURL)
                Toggle("Launch the backend automatically", isOn: $autoLaunch)
                TextField("Backend command", text: $backendCommand)
                    .font(.system(.body, design: .monospaced))
                Text("Runs in a login shell, e.g. `cd ~/hushkey/backend && uv run hushkey serve --llm`.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Reconnect") { Task { await model.connectBackend() } }
                    Button("Restart backend") { model.restartBackend() }
                }
                GroupBox("Backend log") {
                    ScrollView {
                        Text(model.backendLog.suffix(80).joined(separator: "\n"))
                            .font(.system(size: 10, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 140)
                }
            }
            .tabItem { Label("Backend", systemImage: "cpu") }
        }
        .padding(20)
        .frame(width: 560, height: 420)
    }
}
