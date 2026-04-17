require 'json'
package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'CardDetector'
  s.version        = package['version']
  s.summary        = 'Card corner detection using Vision framework and OpenCV'
  s.homepage       = 'https://github.com/RyGaff/mtg-binder-ai'
  s.license        = 'MIT'
  s.authors        = { 'RyGaff' => '' }
  s.platform       = :ios, '15.1'
  s.source         = { :path => '.' }
  s.source_files   = '*.swift', '*.h', '*.mm'
  s.frameworks     = 'Vision', 'CoreMedia', 'CoreImage'
  s.dependency 'ExpoModulesCore'
  s.dependency 'VisionCamera'
end
