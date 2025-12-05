import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  LogBox,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

// --- IMPORTS FOR OFFLINE AI ---
import { useAssets } from 'expo-asset';

// ✅ FIX 1: Keep this "Legacy" import to prevent the FileSystem crash
import * as FileSystem from 'expo-file-system/legacy';

import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import jpeg from 'jpeg-js';
import { loadTensorflowModel } from 'react-native-fast-tflite';

// --- ICONS & STORAGE ---
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

// --- POLYFILL FOR BUFFER ---
import { Buffer } from 'buffer';
global.Buffer = global.Buffer || Buffer;

// Ignore warnings
LogBox.ignoreLogs(['Unable to activate keep awake', 'ImagePicker.MediaTypeOptions']);

const COLORS = {
  primary: '#2E7D32',    
  secondary: '#E8F5E9',  
  accent: '#F57C00',     
  danger: '#C62828',     
  text: '#1F2937',       
  white: '#FFFFFF',
  shadow: '#000',
};

const LABELS = [
  'Bacterial Leaf Blight', 
  'Brown Spot', 
  'Healthy Rice Leaf', 
  'Leaf Blast', 
  'Leaf Scald', 
  'NOT_A_RICE_LEAF',
  'Sheath Blight'
];

const DISEASE_ADVICE: { [key: string]: string } = {
  'Bacterial Leaf Blight': '⚠️ Treatment: Use copper-based sprays. Avoid excessive Nitrogen.',
  'Brown Spot': '⚠️ Treatment: Improve soil fertility (Potassium/Calcium). Treat seeds.',
  'Leaf Blast': '⚠️ Treatment: Apply Tricyclazole. Maintain water level.',
  'Leaf Scald': '⚠️ Treatment: Use clean seeds. Avoid high Nitrogen.',
  'Sheath Blight': '⚠️ Treatment: Apply Azoxystrobin. Reduce plant density.',
  'Healthy Rice Leaf': '✅ Good News: Your crop looks healthy!',
  'NOT_A_RICE_LEAF': '❓ Unknown: Please try a clearer photo.',
};

export default function App() {
  const [assets] = useAssets([require('../../assets/rice_disease_model.tflite')]);
  const [model, setModel] = useState<any>(null);

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (assets && assets[0].localUri) {
      console.log("Loading model from:", assets[0].localUri);
      
      loadTensorflowModel({ url: assets[0].localUri })
        .then((loadedModel) => {
          console.log("Model loaded successfully!");
          setModel(loadedModel);
        })
        .catch((err) => console.error("Failed to load model:", err));
    }
  }, [assets]);

  const handleImageResult = (result: ImagePicker.ImagePickerResult) => {
    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setPrediction(null);
    }
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      // ✅ FIX 2: Reverted to MediaTypeOptions to fix the TypeScript error
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    handleImageResult(result);
  };

  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Permission Refused", "You need to allow camera access.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      // ✅ FIX 2: Reverted to MediaTypeOptions to fix the TypeScript error
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    handleImageResult(result);
  };

  const analyzeImage = async () => {
    if (!model || !selectedImage) {
      Alert.alert("Error", "Model not ready yet.");
      return;
    }
    setLoading(true);

    try {
      // 1. Resize
      const manipResult = await ImageManipulator.manipulateAsync(
        selectedImage,
        [{ resize: { width: 224, height: 224 } }],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      // 2. Read as Base64 (Using the legacy import ensures this works)
      const base64 = await FileSystem.readAsStringAsync(manipResult.uri, {
        encoding: 'base64', 
      });

      const buffer = Buffer.from(base64, 'base64');
      const rawData = jpeg.decode(buffer, { useTArray: true });

      // 3. Convert to Float32
      const float32Data = new Float32Array(224 * 224 * 3);
      let p = 0;
      for (let i = 0; i < rawData.data.length; i += 4) {
        float32Data[p++] = rawData.data[i];     // Red
        float32Data[p++] = rawData.data[i + 1]; // Green
        float32Data[p++] = rawData.data[i + 2]; // Blue
      }

      // 4. Run Inference
      const outputs = await model.run([float32Data]);
      
      const probabilities = outputs[0] as Float32Array;
      
      const probArray = Array.from(probabilities); 
      const maxScore = Math.max(...probArray);
      const maxIndex = probArray.indexOf(maxScore);
      
      const label = LABELS[maxIndex];

      const resultData = {
        label: label,
        confidence: (maxScore * 100).toFixed(2) + '%'
      };

      setPrediction(resultData);
      addToHistory(resultData, selectedImage);

    } catch (error) {
      console.error(error);
      Alert.alert("Error", "Could not analyze image.");
    } finally {
      setLoading(false);
    }
  };

  const addToHistory = async (predictionData: any, imageUri: string) => {
    try {
      const newEntry = {
        id: Date.now().toString(),
        label: predictionData.label,
        confidence: predictionData.confidence,
        imageUri: imageUri,
        date: new Date().toLocaleString(),
        advice: DISEASE_ADVICE[predictionData.label] || "Consult an expert."
      };
      const existingHistory = await AsyncStorage.getItem('leaf_history');
      const historyArray = existingHistory ? JSON.parse(existingHistory) : [];
      const updatedHistory = [newEntry, ...historyArray];
      await AsyncStorage.setItem('leaf_history', JSON.stringify(updatedHistory.slice(0, 20)));
    } catch (error) {
      console.error("Failed to save history:", error);
    }
  };

  if (!model) return (
    <View style={{flex:1, justifyContent:'center', alignItems:'center', backgroundColor: COLORS.secondary}}>
      <ActivityIndicator size="large" color={COLORS.primary} />
      <Text style={{marginTop: 10, color: COLORS.primary, fontWeight:'bold'}}>Loading Rice Doctor AI...</Text>
    </View>
  );

  return (
    <View style={styles.mainContainer}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />
      
      <View style={styles.headerContainer}>
        <MaterialIcons name="grass" size={32} color={COLORS.white} />
        <Text style={styles.headerTitle}>Rice Leaf Doctor</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        
        {!selectedImage && !prediction && (
          <View style={styles.welcomeCard}>
            <Text style={styles.welcomeTitle}>Scan Your Crop</Text>
            <Text style={styles.welcomeText}>
              Take a photo of a rice leaf to detect diseases and get treatment advice instantly (Offline).
            </Text>
          </View>
        )}

        <View style={styles.actionRow}>
          <TouchableOpacity style={styles.actionButton} onPress={pickImage}>
            <Ionicons name="images" size={28} color={COLORS.primary} />
            <Text style={styles.actionText}>Gallery</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionButton} onPress={takePhoto}>
            <Ionicons name="camera" size={28} color={COLORS.primary} />
            <Text style={styles.actionText}>Camera</Text>
          </TouchableOpacity>
        </View>

        {selectedImage && (
          <View style={styles.imageCard}>
            <Text style={styles.sectionLabel}>Original Photo</Text>
            <Image source={{ uri: selectedImage }} style={styles.previewImage} />
            
            <TouchableOpacity 
              style={styles.analyzeButton} 
              onPress={analyzeImage}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color={COLORS.white} />
              ) : (
                <>
                  <MaterialIcons name="search" size={24} color={COLORS.white} />
                  <Text style={styles.analyzeButtonText}>Diagnose Disease</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        )}

        {prediction && (
          <View style={styles.resultCard}>
            <View style={styles.resultHeader}>
              <Text style={styles.resultTitle}>Diagnosis Result</Text>
              <View style={styles.confidenceBadge}>
                <Text style={styles.confidenceText}>{prediction.confidence}</Text>
              </View>
            </View>

            <Text style={styles.diseaseName}>{prediction.label}</Text>

            <View style={styles.adviceBox}>
              <Ionicons name="bulb" size={24} color={COLORS.accent} style={{marginBottom: 5}}/>
              <Text style={styles.adviceText}>
                {DISEASE_ADVICE[prediction.label] || "Please consult an agricultural expert."}
              </Text>
            </View>
          </View>
        )}
        
        <View style={{height: 50}} />
      </ScrollView>
    </View>
  );
}

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  mainContainer: { flex: 1, backgroundColor: '#F8F9FA' },
  headerContainer: {
    paddingTop: 50, paddingBottom: 20, paddingHorizontal: 20,
    backgroundColor: COLORS.primary, flexDirection: 'row', alignItems: 'center',
    borderBottomLeftRadius: 25, borderBottomRightRadius: 25, elevation: 5,
  },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: COLORS.white, marginLeft: 10 },
  scrollContent: { padding: 20, alignItems: 'center' },
  welcomeCard: {
    backgroundColor: COLORS.white, padding: 20, borderRadius: 15, width: '100%',
    marginBottom: 20, elevation: 2, alignItems: 'center',
  },
  welcomeTitle: { fontSize: 18, fontWeight: 'bold', color: COLORS.text, marginBottom: 8 },
  welcomeText: { fontSize: 14, color: '#666', textAlign: 'center', lineHeight: 20 },
  actionRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 20 },
  actionButton: {
    backgroundColor: COLORS.white, width: '48%', paddingVertical: 20, borderRadius: 15,
    alignItems: 'center', justifyContent: 'center', elevation: 3,
    shadowColor: COLORS.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4,
  },
  actionText: { marginTop: 8, fontSize: 16, fontWeight: '600', color: COLORS.primary },
  imageCard: {
    backgroundColor: COLORS.white, borderRadius: 15, padding: 15, width: '100%',
    alignItems: 'center', elevation: 3, marginBottom: 20,
  },
  sectionLabel: { fontSize: 14, fontWeight: '600', color: '#666', alignSelf: 'flex-start', marginBottom: 10 },
  previewImage: { width: width - 80, height: width - 80, borderRadius: 10, backgroundColor: '#eee', resizeMode: 'cover' },
  analyzeButton: {
    backgroundColor: COLORS.primary, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingVertical: 15, paddingHorizontal: 40, borderRadius: 30, marginTop: 20, width: '100%', elevation: 5,
  },
  analyzeButtonText: { color: COLORS.white, fontSize: 18, fontWeight: 'bold', marginLeft: 10 },
  resultCard: {
    backgroundColor: COLORS.white, borderRadius: 15, padding: 20, width: '100%',
    elevation: 4, borderTopWidth: 5, borderTopColor: COLORS.primary,
  },
  resultHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  resultTitle: { fontSize: 14, fontWeight: '600', color: '#888', textTransform: 'uppercase' },
  confidenceBadge: { backgroundColor: COLORS.secondary, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  confidenceText: { color: COLORS.primary, fontWeight: 'bold', fontSize: 12 },
  diseaseName: { fontSize: 24, fontWeight: 'bold', color: COLORS.danger, marginBottom: 15 },
  adviceBox: {
    backgroundColor: '#FFF3E0', padding: 15, borderRadius: 10, borderLeftWidth: 4,
    borderLeftColor: COLORS.accent, marginBottom: 20,
  },
  adviceText: { fontSize: 15, color: '#5D4037', lineHeight: 22 },
});