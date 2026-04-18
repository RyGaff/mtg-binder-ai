// Compilation unit that pulls the shared C++ card detector into the iOS pod.
// CocoaPods strips source_files paths that live outside the pod's root directory,
// so we cannot list ../cpp/card_detector.cpp in source_files directly. Including
// the .cpp here (built as Objective-C++) compiles it alongside the bridge code.
#include "../cpp/card_detector.cpp"
