#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'
require 'xcodeproj'

root = File.expand_path('..', __dir__)
project_path = File.join(root, 'ScyllasBandStudio.xcodeproj')
FileUtils.rm_rf(project_path)

project = Xcodeproj::Project.new(project_path)
project.root_object.attributes['LastSwiftUpdateCheck'] = '2660'
project.root_object.attributes['LastUpgradeCheck'] = '2660'
project.root_object.development_region = 'en'

target = project.new_target(:application, 'ScyllasBandStudio', :ios, '16.0')
source_group = project.main_group.new_group('ScyllasBandStudio', 'ScyllasBandStudio')

Dir.glob(File.join(root, 'ScyllasBandStudio', '*.swift')).sort.each do |absolute_path|
  reference = source_group.new_file(File.basename(absolute_path))
  target.source_build_phase.add_file_reference(reference)
end
source_group.new_file('Info.plist')

asset_phase = target.new_shell_script_build_phase('Prepare Scylla\'s Band assets')
asset_phase.shell_path = '/bin/zsh'
asset_phase.shell_script = <<~'SCRIPT'
  "${SRCROOT}/Scripts/prepare_assets.sh" "${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
SCRIPT
asset_phase.show_env_vars_in_log = '0'
asset_phase.always_out_of_date = '1'

project.build_configurations.each do |configuration|
  configuration.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '16.0'
end

target.build_configurations.each do |configuration|
  settings = configuration.build_settings
  settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'org.scyllasband.studio'
  settings['PRODUCT_NAME'] = '$(TARGET_NAME)'
  settings['INFOPLIST_FILE'] = 'ScyllasBandStudio/Info.plist'
  settings['GENERATE_INFOPLIST_FILE'] = 'NO'
  settings['SWIFT_VERSION'] = '5.0'
  settings['TARGETED_DEVICE_FAMILY'] = '1,2'
  settings['CODE_SIGN_STYLE'] = 'Automatic'
  settings['CURRENT_PROJECT_VERSION'] = '1'
  settings['MARKETING_VERSION'] = '1.0'
  settings['ENABLE_USER_SCRIPT_SANDBOXING'] = 'NO'
  settings['LD_RUNPATH_SEARCH_PATHS'] = '$(inherited) @executable_path/Frameworks'
end

project.save

scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(target)
scheme.set_launch_target(target)
scheme.save_as(project_path, 'ScyllasBandStudio', true)

puts "Generated #{project_path}"
