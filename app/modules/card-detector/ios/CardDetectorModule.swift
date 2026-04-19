import ExpoModulesCore

public class CardDetectorModule: Module {
    public func definition() -> ModuleDefinition {
        Name("CardDetector")

        OnCreate {
            CardDetectorBridge.registerFrameProcessorPlugin()
        }

        AsyncFunction("detectCardCorners") { (uri: String) -> [String: Any]? in
            return CardDetectorBridge.detectCorners(fromFileURI: uri) as? [String: Any]
        }

        AsyncFunction("encodeImage") { (uri: String) -> [NSNumber]? in
            return CardDetectorBridge.encodeImage(fromFileURI: uri)
        }
    }
}
