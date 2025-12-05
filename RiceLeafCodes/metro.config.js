const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Tell the bundler to recognize .tflite files
config.resolver.assetExts.push('tflite');

module.exports = config;