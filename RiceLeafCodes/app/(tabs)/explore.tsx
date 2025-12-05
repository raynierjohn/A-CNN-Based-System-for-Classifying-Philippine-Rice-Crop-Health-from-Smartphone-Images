import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useState } from 'react';
import {
  Alert,
  FlatList,
  Image,
  ListRenderItem,
  Modal,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

// --- TYPESCRIPT INTERFACES ---
interface HistoryItem {
  id: string;
  label: string;
  confidence: string;
  imageUri: string;
  date: string;
  advice: string;
}

// --- THEME COLORS ---
const COLORS = {
  primary: '#2E7D32',
  secondary: '#E8F5E9',
  accent: '#F57C00',
  text: '#1F2937',
  white: '#FFFFFF',
  danger: '#C62828',
  gray: '#6B7280',
  background: '#F8F9FA',
  overlay: 'rgba(0,0,0,0.5)' // Dimmed background for modal
};

export default function ExploreScreen() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  // 1. NEW STATE: Track which item is currently open
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const loadHistory = async () => {
    try {
      const storedHistory = await AsyncStorage.getItem('leaf_history');
      if (storedHistory) {
        setHistory(JSON.parse(storedHistory));
      }
    } catch (error) {
      console.error("Failed to load history", error);
    }
  };

  const clearHistory = async () => {
    Alert.alert(
      "Clear History",
      "Are you sure you want to delete all past records?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Delete", 
          style: "destructive", 
          onPress: async () => {
            await AsyncStorage.removeItem('leaf_history');
            setHistory([]);
          }
        }
      ]
    );
  };

  // 2. MODIFIED: renderItem is now a TouchableOpacity
  const renderItem: ListRenderItem<HistoryItem> = ({ item }) => (
    <TouchableOpacity 
      style={styles.card} 
      onPress={() => setSelectedItem(item)} // Open the modal on click
      activeOpacity={0.7}
    >
      {/* Left Side: Image Thumbnail */}
      <Image source={{ uri: item.imageUri }} style={styles.thumbnail} />

      {/* Right Side: Details */}
      <View style={styles.cardContent}>
        <View style={styles.headerRow}>
          <Text style={styles.dateText}>{item.date}</Text>
          <View style={styles.confidenceBadge}>
            <Text style={styles.confidenceText}>{item.confidence}</Text>
          </View>
        </View>

        <Text style={styles.diseaseTitle}>{item.label}</Text>
        
        <Text style={styles.advicePreview} numberOfLines={2}>
          {item.advice}
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.primary} />
      
      {/* Header Section */}
      <View style={styles.header}>
        <View style={styles.headerTextContainer}>
            <MaterialIcons name="history" size={28} color={COLORS.white} />
            <Text style={styles.headerTitle}>Disease History</Text>
        </View>
        
        {history.length > 0 && (
          <TouchableOpacity onPress={clearHistory} style={styles.deleteButton}>
            <MaterialIcons name="delete-outline" size={24} color={COLORS.white} />
          </TouchableOpacity>
        )}
      </View>

      {/* Main List Area */}
      {history.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Ionicons name="leaf-outline" size={80} color="#ccc" />
          <Text style={styles.emptyText}>No history yet.</Text>
          <Text style={styles.emptySubtext}>
            Scan a rice leaf in the Home screen to see results here.
          </Text>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* 3. NEW: DETAIL POPUP MODAL */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={selectedItem !== null}
        onRequestClose={() => setSelectedItem(null)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContainer}>
            {/* Close Button */}
            <TouchableOpacity 
              style={styles.closeButton} 
              onPress={() => setSelectedItem(null)}
            >
              <Ionicons name="close-circle" size={36} color={COLORS.gray} />
            </TouchableOpacity>

            {selectedItem && (
              <ScrollView showsVerticalScrollIndicator={false}>
                {/* Full Size Image */}
                <Image source={{ uri: selectedItem.imageUri }} style={styles.modalImage} />

                <View style={styles.modalContent}>
                  {/* Title & Date */}
                  <Text style={styles.modalTitle}>{selectedItem.label}</Text>
                  <Text style={styles.modalDate}>{selectedItem.date}</Text>

                  {/* Confidence Badge */}
                  <View style={styles.modalBadge}>
                    <Text style={styles.modalBadgeText}>
                      Confidence: {selectedItem.confidence}
                    </Text>
                  </View>

                  {/* Full Advice */}
                  <View style={styles.adviceBox}>
                    <Text style={styles.adviceLabel}>Treatment & Advice:</Text>
                    <Text style={styles.adviceFullText}>
                      {selectedItem.advice}
                    </Text>
                  </View>
                </View>
              </ScrollView>
            )}
          </View>
        </View>
      </Modal>

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  headerTextContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: COLORS.white,
  },
  deleteButton: {
      padding: 5
  },
  listContent: {
    padding: 15,
  },
  // --- CARD STYLES ---
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 12,
    marginBottom: 15,
    flexDirection: 'row',
    padding: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#eee',
    borderWidth: 1,
    borderColor: '#f0f0f0'
  },
  cardContent: {
    flex: 1,
    marginLeft: 15,
    justifyContent: 'center',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
    alignItems: 'center'
  },
  dateText: {
    fontSize: 12,
    color: COLORS.gray,
  },
  confidenceBadge: {
    backgroundColor: COLORS.secondary,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
  },
  confidenceText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.primary,
  },
  diseaseTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: COLORS.danger,
    marginBottom: 4,
  },
  advicePreview: {
    fontSize: 12,
    color: COLORS.text,
    fontStyle: 'italic',
    lineHeight: 16
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: -50,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.gray,
    marginTop: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 5,
    textAlign: 'center',
    paddingHorizontal: 40
  },

  // --- MODAL STYLES (NEW) ---
  modalOverlay: {
    flex: 1,
    backgroundColor: COLORS.overlay,
    justifyContent: 'flex-end',
  },
  modalContainer: {
    backgroundColor: COLORS.white,
    height: '85%', // Takes up 85% of the screen
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    padding: 20,
    paddingTop: 10,
    elevation: 20,
  },
  closeButton: {
    alignSelf: 'flex-end',
    marginBottom: 10,
    zIndex: 10,
  },
  modalImage: {
    width: '100%',
    height: 300,
    borderRadius: 15,
    marginBottom: 20,
    resizeMode: 'cover',
  },
  modalContent: {
    paddingBottom: 40,
  },
  modalTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: COLORS.danger,
    marginBottom: 5,
  },
  modalDate: {
    fontSize: 14,
    color: COLORS.gray,
    marginBottom: 15,
  },
  modalBadge: {
    backgroundColor: COLORS.secondary,
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginBottom: 20,
  },
  modalBadgeText: {
    color: COLORS.primary,
    fontWeight: 'bold',
    fontSize: 14,
  },
  adviceBox: {
    backgroundColor: '#FFF3E0', // Light Orange
    padding: 15,
    borderRadius: 12,
    borderLeftWidth: 5,
    borderLeftColor: COLORS.accent,
  },
  adviceLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: COLORS.accent,
    marginBottom: 10,
  },
  adviceFullText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  }
});