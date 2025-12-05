import { useAssets } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import jpeg from 'jpeg-js'; // Library to decode JPG to raw pixels
import { useState } from 'react';
import { ActivityIndicator, Alert, Button, Image, ScrollView, Text, View } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';

// Update these with your exact classes from training
const LABELS = [
  'Bacterial Leaf Blight', 
  'Brown Spot', 
  'Healthy Rice Leaf', 
  'Leaf Blast', 
  'Leaf Scald', 
  'NOT_A_RICE_LEAF',
  'Sheath Blight'
];

export default function App() {
  const [modelUri] = useAssets([require('./assets/rice_disease_model.tflite')]);
  const tflite = useTensorflowModel(modelUri ? modelUri[0] : null);
  
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true, // Allows user to crop/center the leaf
      aspect: [1, 1],      // Square aspect ratio
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!tflite.model || !imageUri) return;
    setIsProcessing(true);

    try {
      // 1. Resize image to 224x224 (Required by EfficientNet)
      const manipResult = await ImageManipulator.manipulateAsync(
        imageUri,
        [{ resize: { width: 224, height: 224 } }],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      // 2. Read image as Base64
      const base64 = await FileSystem.readAsStringAsync(manipResult.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // 3. Decode JPG to Raw RGB Bytes
      const buffer = Buffer.from(base64, 'base64');
      const rawData = jpeg.decode(buffer, { useTArray: true }); // Returns Uint8Array

      // 4. Convert to Float32 and Normalize
      // EfficientNet usually expects inputs [0, 255] or [-1, 1] depending on training.
      // Standard Keras EfficientNet expects [0, 255] (Raw pixel values).
      const float32Data = new Float32Array(224 * 224 * 3);
      let p = 0;
      for (let i = 0; i < rawData.data.length; i += 4) {
        float32Data[p++] = rawData.data[i];     // Red
        float32Data[p++] = rawData.data[i + 1]; // Green
        float32Data[p++] = rawData.data[i + 2]; // Blue
        // Ignore [i+3] (Alpha/Transparency)
      }

      // 5. Run Inference
      const outputs = await tflite.model.run([float32Data]);
      
      // 6. Process Output
      const probabilities = outputs[0];
      const maxIndex = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[maxIndex];

      setResult({
        label: LABELS[maxIndex],
        confidence: (confidence * 100).toFixed(2) + '%'
      });

    } catch (error) {
      console.error(error);
      Alert.alert("Error", "Could not analyze image.");
    } finally {
      setIsProcessing(false);
    }
  };

  if (!tflite.model) return (
    <View style={{flex:1, justifyContent:'center', alignItems:'center'}}>
      <ActivityIndicator size="large" />
      <Text>Loading AI Model...</Text>
    </View>
  );

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, alignItems: 'center', justifyContent: 'center', padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold', marginBottom: 20 }}>
        Offline Rice Doctor
      </Text>

      {imageUri ? (
        <Image source={{ uri: imageUri }} style={{ width: 224, height: 224, borderRadius: 10, marginBottom: 20 }} />
      ) : (
        <View style={{ width: 224, height: 224, backgroundColor: '#eee', justifyContent: 'center', alignItems: 'center', marginBottom: 20 }}>
          <Text style={{ color: '#888' }}>No Image Selected</Text>
        </View>
      )}

      <View style={{ flexDirection: 'row', gap: 20 }}>
        <Button title="Pick Image" onPress={pickImage} />
        <Button title="Diagnose" onPress={analyzeImage} disabled={!imageUri || isProcessing} />
      </View>

      {isProcessing && <ActivityIndicator style={{ marginTop: 20 }} size="large" color="blue" />}

      {result && (
        <View style={{ marginTop: 30, alignItems: 'center', padding: 20, backgroundColor: '#e6fffa', borderRadius: 10 }}>
          <Text style={{ fontSize: 22, fontWeight: 'bold', color: '#004d40' }}>{result.label}</Text>
          <Text style={{ fontSize: 16, color: '#00796b' }}>Confidence: {result.confidence}</Text>
        </View>
      )}
    </ScrollView>
  );
}