import AVFoundation
import CoreMedia
import Foundation

/// Webcam capture at ~30 fps. Frames are delivered (un-mirrored, like the training data) on a
/// private serial queue to `onFrame`.
final class CameraService: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum CameraError: LocalizedError {
        case denied, noDevice, configuration(String)
        var errorDescription: String? {
            switch self {
            case .denied: return "Camera access was denied. Enable Hushkey in System Settings ▸ Privacy & Security ▸ Camera."
            case .noDevice: return "No camera found."
            case .configuration(let s): return "Camera configuration failed: \(s)"
            }
        }
    }

    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "hushkey.camera", qos: .userInteractive)
    var onFrame: ((CVPixelBuffer, Double) -> Void)?
    private(set) var isRunning = false

    static var authorization: AVAuthorizationStatus { AVCaptureDevice.authorizationStatus(for: .video) }

    static func requestAccess() async -> Bool {
        switch authorization {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    static func availableCameras() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video, position: .unspecified
        ).devices
    }

    func start(cameraID: String) throws {
        guard Self.authorization == .authorized else { throw CameraError.denied }
        let cams = Self.availableCameras()
        guard let device = cams.first(where: { $0.uniqueID == cameraID }) ?? AVCaptureDevice.default(for: .video) ?? cams.first else {
            throw CameraError.noDevice
        }
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else { throw CameraError.configuration("cannot add input") }
            session.addInput(input)
        } catch let e as CameraError {
            session.commitConfiguration(); throw e
        } catch {
            session.commitConfiguration(); throw CameraError.configuration(error.localizedDescription)
        }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { session.commitConfiguration(); throw CameraError.configuration("cannot add output") }
        session.addOutput(output)
        if let conn = output.connection(with: .video), conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = false  // the model was trained on un-mirrored video
        }
        session.commitConfiguration()
        try? device.lockForConfiguration()
        if device.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 30 }) {
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        }
        device.unlockForConfiguration()
        queue.async { [weak self] in self?.session.startRunning() }
        isRunning = true
    }

    func stop() {
        queue.async { [weak self] in self?.session.stopRunning() }
        isRunning = false
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(pb, t)
    }
}
