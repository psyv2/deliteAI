import Foundation
@objc public class EspeakNGCallbacks: NSObject {
    @objc public static var initializeEspeakCallback: (() -> Int)?
    @objc public static var espeakTextToPhonemesCallback: ((String) -> String)?
}
