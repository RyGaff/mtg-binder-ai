require 'json'
package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'CardDetector'
  s.version        = package['version']
  s.summary        = 'Card corner detection using OpenCV'
  s.homepage       = 'https://github.com/RyGaff/mtg-binder-ai'
  s.license        = 'MIT'
  s.authors        = { 'RyGaff' => '' }
  s.platform       = :ios, '15.1'
  s.source         = { :path => '.' }
  s.source_files   = '*.swift', '*.h', '*.mm'
  s.resources      = '*.mlmodel'
  s.frameworks     = 'CoreMedia', 'CoreImage', 'UIKit'
  s.weak_frameworks = 'CoreML'
  s.dependency 'ExpoModulesCore'
  s.dependency 'VisionCamera'
  s.dependency 'OpenCV-Dynamic-Framework', '~> 4.10.0'
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'OTHER_CPLUSPLUSFLAGS' => '-fexceptions -frtti',
    # $(inherited) preserves OpenCV2 + CocoaPods-provided paths
    'HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_TARGET_SRCROOT}/../cpp"',
  }
end
