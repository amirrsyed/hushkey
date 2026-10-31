import CoreImage
import CoreVideo
import Foundation
import ImageIO
import Vision

/// Per-frame work on the camera queue: face landmarks → keypoints, a face+neck crop encoded
/// as grayscale JPEG for the backend, and a small mouth preview for the HUD.
final class FrameProcessor {
    struct Output {
        let t: Double
        let keypoints: FaceKeypoints?  // full-frame coordinates
        let wire: [String: Any]?  // frame message for the backend (only when `encode` is true)
        let mouthPreview: CGImage?
    }

    /// Longest side of the crop sent to the backend. ~320 px keeps the mouth at LRS3-like resolution.
    var maxCropWidth: CGFloat = 320
    var jpegQuality: CGFloat = 0.8

    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let gray = CGColorSpaceCreateDeviceGray()
    private let sequence = VNSequenceRequestHandler()
    private let request: VNDetectFaceLandmarksRequest = {
        let r = VNDetectFaceLandmarksRequest()
        r.revision = VNDetectFaceLandmarksRequestRevision3
        return r
    }()
    private var lastFaceRect: CGRect?
    private var previewCounter = 0

    func process(_ pixelBuffer: CVPixelBuffer, time: Double, encode: Bool, wantPreview: Bool) -> Output {
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        let size = CGSize(width: w, height: h)
        try? sequence.perform([request], on: pixelBuffer, orientation: .up)
        let face = (request.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width })
        let kp = face.flatMap { FaceKeypoints(observation: $0, imageSize: size) }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        var wire: [String: Any]? = nil
        if encode {
            wire = encodeFrame(image: image, size: size, keypoints: kp, time: time)
        }
        var preview: CGImage? = nil
        previewCounter += 1
        if wantPreview, let kp, previewCounter % 3 == 0 {
            preview = mouthPreview(image: image, size: size, keypoints: kp)
        }
        return Output(t: time, keypoints: kp, wire: wire, mouthPreview: preview)
    }

    /// Axis-aligned crop covering eyes → below the chin (so the backend can see the throat).
    private func cropRect(size: CGSize, keypoints kp: FaceKeypoints?) -> CGRect {
        guard let kp, let a = kp.points["right_eye"], let b = kp.points["left_eye"], let chin = kp.points["chin"] else {
            return lastFaceRect ?? CGRect(origin: .zero, size: size)
        }
        let iod = max(kp.interOcular, 10)
        let cx = (a.x + b.x) / 2
        let top = min(a.y, b.y) - 1.0 * iod
        let bottom = chin.y + 1.5 * iod
        var r = CGRect(x: cx - 1.7 * iod, y: top, width: 3.4 * iod, height: bottom - top)
        r = r.intersection(CGRect(origin: .zero, size: size)).integral
        if r.isNull || r.width < 16 || r.height < 16 { r = CGRect(origin: .zero, size: size) }
        lastFaceRect = r
        return r
    }

    private func encodeFrame(image: CIImage, size: CGSize, keypoints: FaceKeypoints?, time: Double) -> [String: Any]? {
        let rect = cropRect(size: size, keypoints: keypoints)
        let scale = min(1, maxCropWidth / rect.width)
        // CIImage uses a bottom-left origin.
        let ciRect = CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height)
        let cropped = image.cropped(to: ciRect)
            .transformed(by: CGAffineTransform(translationX: -ciRect.minX, y: -ciRect.minY))
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let jpeg = ciContext.jpegRepresentation(
            of: cropped, colorSpace: gray,
            options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): jpegQuality]
        ) else { return nil }
        var msg: [String: Any] = ["type": "frame", "t": time, "jpeg": jpeg.base64EncodedString()]
        if let kp = keypoints {
            msg["kp"] = kp.translated(by: rect.origin, scale: scale).json()
        } else {
            msg["kp"] = NSNull()
        }
        return msg
    }

    private func mouthPreview(image: CIImage, size: CGSize, keypoints kp: FaceKeypoints) -> CGImage? {
        guard let m = kp.points["mouth_center"] else { return nil }
        let side = max(kp.interOcular * 1.4, 32)
        let rect = CGRect(x: m.x - side / 2, y: size.height - m.y - side / 2, width: side, height: side)
            .intersection(image.extent)
        guard !rect.isNull else { return nil }
        let img = image.cropped(to: rect).applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
        return ciContext.createCGImage(img, from: rect)
    }
}
