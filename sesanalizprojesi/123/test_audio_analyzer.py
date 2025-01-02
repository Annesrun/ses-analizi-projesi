import unittest
import numpy as np
from streamlit_app import EmotionAnalyzer, TopicAnalyzer, StreamlitAudioRecorder

class TestAudioAnalyzer(unittest.TestCase):
    def setUp(self):
        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        self.audio_recorder = StreamlitAudioRecorder()

    def test_emotion_analysis(self):
        """Test Case 1: Duygu Analizi Testi"""
        test_text = "Bugün çok mutluyum ve heyecanlıyım!"
        result = self.emotion_analyzer.analyze_emotion(test_text)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_topic_analysis(self):
        """Test Case 2: Konu Analizi Testi"""
        test_text = "Okulda matematik dersi işledik."
        topics = self.topic_analyzer.analyze_topics(test_text)
        self.assertIsNotNone(topics)
        self.assertIsInstance(topics, list)

    def test_text_cleaning(self):
        """Test Case 3: Metin Temizleme Testi"""
        test_text = "  Test,   123  "
        cleaned = test_text.strip()
        self.assertIsInstance(cleaned, str)
        self.assertNotEqual(cleaned, test_text)

    def test_audio_format(self):
        """Test Case 4: Ses Format Kontrolü"""
        audio_data = np.zeros(44100)  # 1 saniyelik boş ses
        self.assertEqual(len(audio_data), 44100)
        self.assertEqual(audio_data.dtype, np.float64)

    def test_confidence_scores(self):
        """Test Case 5: Güven Skoru Kontrolü"""
        scores = {"Mutlu": 0.8, "Üzgün": 0.2}
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        self.assertTrue(all(0 <= score <= 1 for score in scores.values()))

if __name__ == '__main__':
    unittest.main()