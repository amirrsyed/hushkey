import Foundation

/// Hands-free segmentation from mouth/jaw motion, adapted from Subvox's `CaptureController`
/// (energy over a calibrated noise floor, with pre-roll and hang-over). Silent speech has
/// longer pauses than voiced speech, so the hang-over is longer.
final class MotionGate {
    static let calibrationSeconds = 2.0
    static let prerollSeconds = 0.4
    static let hangoverSeconds = 0.9
    static let minSeconds = 0.6
    static let maxSeconds = 12.0

    var triggerRatio = 3.0
    private(set) var energy = 0.0
    private(set) var noiseFloor: Double?
    private(set) var isOpen = false

    private var last: (aperture: Double, jaw: Double)?
    private var calibration: [Double] = []
    private var calibrationEnd: Double?
    private var openedAt = 0.0
    private var quietSince: Double?

    var isCalibrated: Bool { noiseFloor != nil }

    func calibrate() {
        noiseFloor = nil
        calibration = []
        calibrationEnd = nil
    }

    enum Event { case none, open(prerollStart: Double), close, discard }

    /// Feed one frame's articulation measurements; returns segment boundaries.
    func ingest(t: Double, articulation a: (aperture: Double, jaw: Double)?) -> Event {
        guard let a else { return isOpen ? closeIfQuiet(t: t, e: 0) : .none }
        let e = last.map { abs(a.aperture - $0.aperture) + 0.5 * abs(a.jaw - $0.jaw) } ?? 0
        last = a
        energy = energy * 0.7 + e * 0.3

        if noiseFloor == nil {
            let end = calibrationEnd ?? t + Self.calibrationSeconds
            calibrationEnd = end
            calibration.append(energy)
            if t >= end {
                let s = calibration.sorted()
                noiseFloor = max(s[s.count / 2], 1e-4)
            }
            return .none
        }
        let threshold = noiseFloor! * triggerRatio
        if !isOpen {
            if energy > threshold {
                isOpen = true
                openedAt = t
                quietSince = nil
                return .open(prerollStart: t - Self.prerollSeconds)
            }
            return .none
        }
        return closeIfQuiet(t: t, e: energy)
    }

    private func closeIfQuiet(t: Double, e: Double) -> Event {
        let threshold = (noiseFloor ?? 1) * triggerRatio
        if e < threshold * 0.7 {
            let since = quietSince ?? t
            quietSince = since
            if t - since > Self.hangoverSeconds { return close(t: t) }
        } else {
            quietSince = nil
        }
        if t - openedAt > Self.maxSeconds { return close(t: t) }
        return .none
    }

    private func close(t: Double) -> Event {
        isOpen = false
        quietSince = nil
        return (t - openedAt >= Self.minSeconds) ? .close : .discard
    }

    func reset() {
        isOpen = false
        quietSince = nil
        last = nil
    }
}
