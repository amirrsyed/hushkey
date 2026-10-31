import AppKit
import SwiftUI

/// Floating, non-activating pill at the bottom of the screen (never steals focus from the
/// app you're dictating into).
@MainActor
final class HUDController {
    private let panel: NSPanel
    private var hideTask: Task<Void, Never>?
    private weak var model: AppModel?

    init(model: AppModel) {
        self.model = model
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 76),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered, defer: true
        )
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        let host = NSHostingView(rootView: HUDView().environment(model))
        host.frame = panel.contentRect(forFrameRect: panel.frame)
        panel.contentView = host
    }

    func show() {
        hideTask?.cancel()
        model?.pipeline.setPreviewWanted(true)
        if let screen = NSScreen.main {
            let f = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: f.midX - panel.frame.width / 2, y: f.minY + 48))
        }
        panel.orderFrontRegardless()
    }

    func hide(after seconds: Double) {
        hideTask?.cancel()
        hideTask = Task { @MainActor [weak self] in
            if seconds > 0 { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
            guard let self, !Task.isCancelled else { return }
            if let m = self.model, m.isBusy { return }
            self.panel.orderOut(nil)
            self.model?.pipeline.setPreviewWanted(self.model?.enrollmentPrompt != nil)
        }
    }
}

struct HUDView: View {
    @Environment(AppModel.self) private var model
    @AppStorage(AppSettings.Key.showHUDPreview) private var showPreview = true

    var body: some View {
        HStack(spacing: 12) {
            if showPreview {
                MouthPreview(image: model.mouthPreview, faceVisible: model.faceVisible)
                    .frame(width: 52, height: 52)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    StatusDot(phase: model.phase, locked: model.handsFreeLocked)
                    Text(title).font(.system(size: 12, weight: .semibold)).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    MotionMeter(level: model.phase == .listening ? model.motionLevel : 0)
                        .frame(width: 60, height: 6)
                }
                Text(bodyText)
                    .font(.system(size: 14))
                    .lineLimit(2)
                    .truncationMode(.head)
                    .foregroundStyle(isError ? Color.red : Color.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(width: 460, height: 76)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).strokeBorder(.white.opacity(0.08)))
    }

    private var isError: Bool { if case .error = model.phase { return true } else { return false } }

    private var title: String {
        switch model.phase {
        case .listening: return model.handsFreeLocked ? "Hands-free — tap to finish" : (model.enrollmentPrompt != nil ? "Recording sample" : "Listening with your lips")
        case .transcribing: return "Reading…"
        case .idle: return "Done"
        case .error: return "Hushkey"
        }
    }

    private var bodyText: String {
        switch model.phase {
        case .listening:
            if !model.faceVisible { return "Can't see your face — look at the camera" }
            return model.partial.isEmpty ? "Mouth your words, no sound needed…" : model.partial.lowercased()
        case .transcribing: return model.partial.isEmpty ? "…" : model.partial.lowercased()
        case .idle: return model.lastText
        case .error(let m): return m
        }
    }
}

struct MouthPreview: View {
    let image: CGImage?
    let faceVisible: Bool
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.black.opacity(0.6))
            if let image, faceVisible {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .scaledToFill()
                    .scaleEffect(x: -1, y: 1)  // mirror for the user, like a selfie view
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            } else {
                Image(systemName: "face.dashed").font(.system(size: 22)).foregroundStyle(.secondary)
            }
        }
    }
}

struct StatusDot: View {
    let phase: AppModel.Phase
    let locked: Bool
    @State private var pulse = false
    var body: some View {
        Group {
            switch phase {
            case .listening:
                Circle().fill(locked ? Color.orange : Color.red)
                    .frame(width: 8, height: 8)
                    .opacity(pulse ? 0.35 : 1)
                    .animation(.easeInOut(duration: 0.6).repeatForever(), value: pulse)
                    .onAppear { pulse = true }
            case .transcribing:
                ProgressView().controlSize(.mini)
            case .idle:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.system(size: 10))
            case .error:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow).font(.system(size: 10))
            }
        }
        .frame(width: 12, height: 12)
    }
}

struct MotionMeter: View {
    let level: Double
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(.secondary.opacity(0.2))
                Capsule().fill(.tint).frame(width: g.size.width * CGFloat(max(0, min(1, level))))
            }
        }
        .animation(.linear(duration: 0.08), value: level)
        .help("Mouth / jaw motion")
    }
}
