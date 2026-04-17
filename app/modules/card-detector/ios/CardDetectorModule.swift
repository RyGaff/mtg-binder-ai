import ExpoModulesCore

public class CardDetectorModule: Module {
    public func definition() -> ModuleDefinition {
        Name("CardDetector")

        AsyncFunction("detectCardCorners") { (uri: String) -> [String: Double]? in
            guard let result = CardDetectorBridge.detectCorners(fromFileURI: uri) as? [String: Double] else { return nil }
            return result
        }
    }
}
