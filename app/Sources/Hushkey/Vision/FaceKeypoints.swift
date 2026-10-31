import CoreGraphics
import Foundation
import Vision

/// The 11 named keypoints shared with the Python backend (see backend/hushkey/preprocess/keypoints.py).
/// Coordinates are pixels with a top-left origin, in whatever image they are attached to.
struct FaceKeypoints {
    static let names = [
        "right_eye", "left_eye", "nose_tip", "mouth_center",
        "lip_upper_inner", "lip_lower_inner", "lip_upper_outer", "lip_lower_outer",
        "lip_left", "lip_right", "chin",
    ]

    var points: [String: CGPoint]

    var interOcular: CGFloat {
        guard let a = points["right_eye"], let b = points["left_eye"] else { return 0 }
        return hypot(b.x - a.x, b.y - a.y)
    }

    /// Lip aperture and jaw opening in inter-ocular units (for the hands-free motion gate).
    var articulation: (aperture: Double, jaw: Double) {
        let d = max(interOcular, 1)
        func dist(_ a: String, _ b: String) -> Double {
            guard let p = points[a], let q = points[b] else { return 0 }
            return Double(hypot(p.x - q.x, p.y - q.y) / d)
        }
        return (dist("lip_upper_inner", "lip_lower_inner"), dist("nose_tip", "chin"))
    }

    func translated(by origin: CGPoint, scale: CGFloat) -> FaceKeypoints {
        var out: [String: CGPoint] = [:]
        for (k, p) in points { out[k] = CGPoint(x: (p.x - origin.x) * scale, y: (p.y - origin.y) * scale) }
        return FaceKeypoints(points: out)
    }

    func json() -> [String: [Double]] {
        var out: [String: [Double]] = [:]
        for (k, p) in points { out[k] = [Double(p.x), Double(p.y)] }
        return out
    }

    /// Reduce Apple Vision's 76-point landmarks to our 11 keypoints.
    init?(observation obs: VNFaceObservation, imageSize size: CGSize) {
        guard let lm = obs.landmarks else { return nil }
        func pts(_ r: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            // Vision's image coordinates have a bottom-left origin; flip to top-left.
            (r?.pointsInImage(imageSize: size) ?? []).map { CGPoint(x: $0.x, y: size.height - $0.y) }
        }
        func centroid(_ p: [CGPoint]) -> CGPoint? {
            guard !p.isEmpty else { return nil }
            let s = p.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            return CGPoint(x: s.x / CGFloat(p.count), y: s.y / CGFloat(p.count))
        }
        guard var eyeA = centroid(pts(lm.rightEye)), var eyeB = centroid(pts(lm.leftEye)),
              let nose = centroid(pts(lm.nose)) else { return nil }
        let outer = pts(lm.outerLips), inner = pts(lm.innerLips), contour = pts(lm.faceContour)
        guard outer.count >= 4, let mouth = centroid(outer), !contour.isEmpty else { return nil }
        if eyeA.x > eyeB.x { swap(&eyeA, &eyeB) }  // image-left eye first, regardless of mirroring

        // Face-aligned axes: u along the eye line, v down the face.
        let ex = CGVector(dx: eyeB.x - eyeA.x, dy: eyeB.y - eyeA.y)
        let len = max(hypot(ex.dx, ex.dy), 1)
        let ux = CGVector(dx: ex.dx / len, dy: ex.dy / len)
        let vy = CGVector(dx: -ux.dy, dy: ux.dx)
        func u(_ p: CGPoint) -> CGFloat { (p.x - mouth.x) * ux.dx + (p.y - mouth.y) * ux.dy }
        func v(_ p: CGPoint) -> CGFloat { (p.x - mouth.x) * vy.dx + (p.y - mouth.y) * vy.dy }

        guard let left = outer.max(by: { u($0) < u($1) }), let right = outer.min(by: { u($0) < u($1) }) else { return nil }
        let halfWidth = max(abs(u(left) - u(right)) / 2, 1)
        func midline(_ p: [CGPoint]) -> [CGPoint] { p.filter { abs(u($0)) < 0.35 * halfWidth } }
        let outerMid = midline(outer).isEmpty ? outer : midline(outer)
        let innerSrc = inner.isEmpty ? outer : inner
        let innerMid = midline(innerSrc).isEmpty ? innerSrc : midline(innerSrc)

        guard let upOut = outerMid.min(by: { v($0) < v($1) }), let loOut = outerMid.max(by: { v($0) < v($1) }),
              let upIn = innerMid.min(by: { v($0) < v($1) }), let loIn = innerMid.max(by: { v($0) < v($1) }),
              let chin = contour.max(by: { v($0) < v($1) }) else { return nil }

        points = [
            "right_eye": eyeA, "left_eye": eyeB, "nose_tip": nose, "mouth_center": mouth,
            "lip_upper_inner": upIn, "lip_lower_inner": loIn,
            "lip_upper_outer": upOut, "lip_lower_outer": loOut,
            "lip_left": left, "lip_right": right, "chin": chin,
        ]
    }

    init(points: [String: CGPoint]) { self.points = points }
}
