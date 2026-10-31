import SwiftUI

/// "Train on my mouth": record silent prompts → personal LoRA adapter.
struct EnrollmentView: View {
    @Environment(AppModel.self) private var model
    @State private var index = 0
    @State private var savedOnDisk = 0
    @State private var pressing = false

    private var prompt: String { EnrollmentPrompts.all[index % EnrollmentPrompts.all.count] }
    private static let recommended = 40

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Train on my mouth").font(.title2.bold())
            Text("Silently mouth each sentence the way you would when dictating. No sound. Around \(Self.recommended) clips (≈5 minutes) is enough for a personal adapter; more helps.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(alignment: .top, spacing: 16) {
                MouthPreview(image: model.mouthPreview, faceVisible: model.faceVisible)
                    .frame(width: 120, height: 120)
                VStack(alignment: .leading, spacing: 10) {
                    Text("Prompt \(index + 1)").font(.caption).foregroundStyle(.secondary)
                    Text(prompt).font(.system(size: 22, weight: .medium)).fixedSize(horizontal: false, vertical: true)
                    if let last = model.lastEnrollment {
                        Text("Model read: “\(last.heard.lowercased())”").font(.callout).foregroundStyle(.secondary)
                    }
                }
            }

            HStack(spacing: 12) {
                recordButton
                Button("Skip") { advance() }
                Spacer()
                Text("\(savedOnDisk) clips saved").monospacedDigit().foregroundStyle(.secondary)
            }

            ProgressView(value: Double(min(savedOnDisk, Self.recommended)), total: Double(Self.recommended))

            Divider()
            HStack {
                Button(model.isTraining ? "Training…" : "Train personal adapter") { model.trainPersonalAdapter() }
                    .disabled(model.isTraining || savedOnDisk < 10)
                Text("Runs `hushkey train personalize` locally (a few minutes on Apple Silicon).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !model.trainingLog.isEmpty {
                ScrollView {
                    Text(model.trainingLog.suffix(60).joined(separator: "\n"))
                        .font(.system(size: 10, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 120)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(24)
        .frame(width: 620)
        .onAppear {
            index = UserDefaults.standard.integer(forKey: "enrollIndex")
            model.openEnrollment(prompt: prompt)
            savedOnDisk = EnrollmentPrompts.savedCount()
        }
        .onDisappear { model.closeEnrollment() }
        .onChange(of: model.enrolledThisSession) { _, _ in
            savedOnDisk = EnrollmentPrompts.savedCount()
            advance()
        }
    }

    private var recordButton: some View {
        Label(pressing ? "Recording… release when done" : "Hold to record (or hold \(AppSettings.hotkey.label))",
              systemImage: pressing ? "record.circle.fill" : "record.circle")
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(pressing ? Color.red.opacity(0.85) : Color.accentColor, in: Capsule())
            .foregroundStyle(.white)
            .onLongPressGesture(minimumDuration: 60, maximumDistance: 80, perform: {}, onPressingChanged: { isPressing in
                pressing = isPressing
                if isPressing { model.beginDictation() } else { model.endDictation() }
            })
            .disabled(!model.backendConnected)
    }

    private func advance() {
        index = (index + 1) % EnrollmentPrompts.all.count
        UserDefaults.standard.set(index, forKey: "enrollIndex")
        model.enrollmentPrompt = prompt
    }
}

enum EnrollmentPrompts {
    static func savedCount() -> Int {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Hushkey/enrollment/manifest.jsonl")
        guard let s = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return s.split(separator: "\n").count
    }

    /// Short, dictation-style sentences written for Hushkey (public domain / CC0).
    static let all: [String] = [
        "I'll be there in ten minutes",
        "Can we move the meeting to Thursday",
        "Thanks for sending this over",
        "Let me know what you think",
        "I'm running a little late today",
        "Please review the attached draft",
        "That sounds good to me",
        "What time works best for you",
        "I need to pick up groceries after work",
        "Remind me to call my mother tonight",
        "The report is almost finished",
        "Could you share the slides from yesterday",
        "I completely agree with your point",
        "Let's grab lunch next week",
        "Happy birthday, hope you have a great day",
        "I'm working from home this morning",
        "Send me the address when you can",
        "We should test this before Friday",
        "Sorry for the slow reply",
        "The build is failing on the main branch",
        "Open a new issue for this bug",
        "I'll follow up with the design team",
        "How was your weekend",
        "Please add me to the calendar invite",
        "This looks great, nice work",
        "I can't make it tonight, maybe tomorrow",
        "Book a table for four at seven",
        "Turn the lights off in the kitchen",
        "Search for coffee shops near me",
        "My flight lands at six thirty",
        "We are out of milk and bread",
        "Give me five minutes to finish this",
        "Let's schedule a quick call",
        "The numbers look better this month",
        "I'm happy to help with that",
        "Can you explain the second step again",
        "Write that down so we don't forget",
        "Meet me by the front entrance",
        "I think the deadline is too tight",
        "Please keep this between us for now",
        "The weather is perfect for a walk",
        "Pause the music and read my messages",
        "Delete the last sentence",
        "Start a new paragraph",
        "Thank you so much for your patience",
        "I'll send the invoice by the end of the day",
        "Does anyone have questions before we start",
        "Let's revisit this in a couple of weeks",
        "My phone battery is almost dead",
        "Good morning, everyone",
    ]
}
