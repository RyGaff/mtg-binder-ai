import VisionCamera

@objc(VisionCameraPlugin_detectCardCornersInFrame)
public class CardDetectorFrameProcessorPlugin: FrameProcessorPlugin {
    public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any? {
        let buffer = frame.buffer
        return CardDetectorBridge.detectCorners(from: buffer)
    }
}
