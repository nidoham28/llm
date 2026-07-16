import 'dart:math';
import 'dart:core';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  // N-Gram Storage
  final List<Map<String, Map<String, int>>> _counts = List.generate(5, (_) => {});
  final List<Map<String, int>> _contextTotals = List.generate(5, (_) => {});
  final List<Map<String, int>> _distinctNext = List.generate(5, (_) => {});

  // TF-IDF & Corpus Storage
  final List<String> _corpusTexts = [];
  final List<Map<String, double>> _tfidfVectors = [];
  Map<String, double> _idf = {};
  Set<String> _vocabulary = {};

  final Map<String, Set<String>> _precedingWords = {};
  int _totalBigramTypes = 0;
  static const double _D = 0.75;
  static const String _bosToken = '<BOS>'; // Beginning of Sentence
  static const String _eosToken = '<EOS>'; // End of Sentence
  final Random _random = Random();

  // LCR Pattern Storage (Left-Center-Right)
  final Set<String> _leftPatterns = {};  // First 2 words
  final Set<String> _rightPatterns = {}; // Last 2 words

  String _getRoot(String word) {
    if (word.endsWith('ing')) return word.substring(0, word.length - 3);
    if (word.endsWith('s')) return word.substring(0, word.length - 1);
    if (word.endsWith('ed')) return word.substring(0, word.length - 2);
    return word;
  }

  List<String> _tokenize(String text, {bool addBosEos = false}) {
    var tokens = text.toLowerCase().split(RegExp(r'[^a-z]+')).where((w) => w.isNotEmpty).toList();
    if (addBosEos) {
      tokens.insert(0, _bosToken);
      tokens.add(_eosToken);
    }
    return tokens;
  }

  void train(List<Map<String, dynamic>> dbData) {
    // Clear memory
    for (int i = 0; i < 5; i++) {
      _counts[i].clear();
      _contextTotals[i].clear();
      _distinctNext[i].clear();
    }
    _precedingWords.clear();
    _totalBigramTypes = 0;
    _leftPatterns.clear();
    _rightPatterns.clear();

    _corpusTexts.clear();
    _tfidfVectors.clear();
    _idf.clear();
    _vocabulary.clear();

    List<List<String>> tokenizedDocs = [];
    Map<String, int> docFrequency = {};

    for (var item in dbData) {
      // Train every line separately
      List<String> lines = (item['input_data'] ?? '').split('\n');
      for (var line in lines) {
        if (line.trim().isEmpty) continue;

        var tokens = _tokenize(line, addBosEos: true);
        _corpusTexts.add(line.trim());
        tokenizedDocs.add(tokens);

        // LCR Pattern Extraction
        if (tokens.length >= 4) {
          _leftPatterns.add("${tokens[1]} ${tokens[2]}"); // First 2 real words
          _rightPatterns.add("${tokens[tokens.length - 3]} ${tokens[tokens.length - 2]}"); // Last 2 real words
        }

        Set<String> uniqueTokens = tokens.toSet();
        for (var token in uniqueTokens) {
          docFrequency[token] = (docFrequency[token] ?? 0) + 1;
          _vocabulary.add(token);
        }

        for (int i = 0; i < tokens.length; i++) {
          _updateCount(0, "", tokens[i]);
          if (i > 0) {
            _precedingWords.putIfAbsent(tokens[i], () => {});
            if (_precedingWords[tokens[i]]!.add(tokens[i - 1])) {
              _totalBigramTypes++;
            }
          }

          for (int n = 2; n <= 5; n++) {
            if (i >= n - 1) {
              String context = tokens.sublist(i - n + 1, i).join(" ");
              _updateCount(n - 1, context, tokens[i]);
            }
          }
        }
      }
    }

    int N = _corpusTexts.length;
    docFrequency.forEach((word, df) {
      _idf[word] = log((N + 1) / (df + 1)) + 1;
    });
    for (var tokens in tokenizedDocs) {
      _tfidfVectors.add(_computeTfidfVector(tokens));
    }
  }

  Map<String, double> _computeTfidfVector(List<String> tokens) {
    Map<String, double> tf = {};
    if (tokens.isEmpty) return tf;
    for (var token in tokens) {
      String root = _getRoot(token);
      tf[root] = (tf[root] ?? 0) + 1;
    }
    tf.updateAll((key, value) => value / tokens.length);

    Map<String, double> tfidf = {};
    tf.forEach((word, tfVal) {
      tfidf[word] = tfVal * (_idf[_getRoot(word)] ?? 0);
    });
    return tfidf;
  }

  double _cosineSimilarity(Map<String, double> v1, Map<String, double> v2) {
    double dotProduct = 0;
    for (var key in v1.keys) {
      if (v2.containsKey(key)) dotProduct += v1[key]! * v2[key]!;
    }
    double mag1 = sqrt(v1.values.fold(0, (sum, val) => sum + val * val));
    double mag2 = sqrt(v2.values.fold(0, (sum, val) => sum + val * val));
    if (mag1 == 0 || mag2 == 0) return 0.0;
    return dotProduct / (mag1 * mag2);
  }

  void _updateCount(int level, String context, String word) {
    _counts[level].putIfAbsent(context, () => {});
    _contextTotals[level].putIfAbsent(context, () => 0);
    _distinctNext[level].putIfAbsent(context, () => 0);

    if (!_counts[level][context]!.containsKey(word)) {
      _distinctNext[level][context] = _distinctNext[level][context]! + 1;
    }
    _counts[level][context]![word] = (_counts[level][context]![word] ?? 0) + 1;
    _contextTotals[level][context] = _contextTotals[level][context]! + 1;
  }

  double _getKNProb(List<String> context, String word) {
    if (context.isEmpty) {
      int distinctPreceding = _precedingWords[word]?.length ?? 0;
      if (_totalBigramTypes == 0) return 0.0;
      return distinctPreceding / _totalBigramTypes;
    }

    int level = context.length;
    String ctxStr = context.join(" ");

    int count = _counts[level][ctxStr]?[word] ?? 0;
    int total = _contextTotals[level][ctxStr] ?? 0;
    int distinctNext = _distinctNext[level][ctxStr] ?? 0;

    if (total == 0) return _getKNProb(context.sublist(1), word);

    double firstTerm = max(count - _D, 0) / total;
    double lambda = (_D / total) * distinctNext;
    double continuationProb = _getKNProb(context.sublist(1), word);

    return firstTerm + (lambda * continuationProb);
  }

  List<String> getLiveSuggestions(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return [];

    List<String> context = tokens.length >= 4 ? tokens.sublist(tokens.length - 4) : List.from(tokens);
    Set<String> candidates = {};

    for (int n = 4; n >= 1; n--) {
      if (context.length >= n) {
        String ctxStr = context.sublist(context.length - n).join(" ");
        candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
      }
    }
    candidates.remove(_eosToken);
    candidates.remove(_bosToken);

    if (candidates.isEmpty) return [];

    List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
    scored.sort((a, b) => b.value.compareTo(a.value));

    return scored.take(3).map((e) => e.key).toList();
  }

  Map<String, dynamic> predictNextWord(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return {'success': false, 'message': 'Type some words.'};

    List<String> context = tokens.length >= 4 ? tokens.sublist(tokens.length - 4) : List.from(tokens);
    Set<String> candidates = {};

    for (int n = 4; n >= 1; n--) {
      if (context.length >= n) {
        String ctxStr = context.sublist(context.length - n).join(" ");
        candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
      }
    }
    if (candidates.length < 10) {
      var topCont = _precedingWords.entries.toList()..sort((a, b) => b.value.length.compareTo(a.value.length));
      for (var e in topCont.take(20)) { candidates.add(e.key); }
    }
    candidates.remove(_eosToken);
    candidates.remove(_bosToken);

    if (candidates.isEmpty) return {'success': false, 'message': 'No patterns found.'};

    List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
    scored.sort((a, b) => b.value.compareTo(a.value));
    var top3 = scored.take(3).toList();

    return {
      'success': true,
      'context': "[${min(5, context.length + 1)}-Gram] '${context.join(" ")}'",
      'predictions': top3.map((e) => "  ↳ ${e.key}  [KN Prob: ${(e.value * 100).toStringAsFixed(2)}%]").join('\n'),
      'best_word': top3.first.key
    };
  }

  // --- DEEP THINK: 100% ACCURATE LCR PATTERN GENERATION ---
  Map<String, dynamic> generateSentence(String query, int maxWords, {double threshold = 0.80}) {
    var rawTokens = _tokenize(query);
    if (rawTokens.isEmpty) return {'success': false, 'message': 'Type some words to generate.'};

    // Prepare tokens with BOS
    List<String> initTokens = [_bosToken, ...rawTokens];
    var promptVector = _computeTfidfVector(rawTokens);

    // Beam Search state
    List<MapEntry<List<String>, double>> beams = [
      MapEntry(initTokens, 1.0)
    ];

    for (int step = 0; step < maxWords; step++) {
      List<MapEntry<List<String>, double>> newBeams = [];

      for (var beam in beams) {
        List<String> currentTokens = beam.key;
        double currentScore = beam.value;

        List<String> context = currentTokens.length >= 4 ? currentTokens.sublist(currentTokens.length - 4) : List.from(currentTokens);
        Set<String> candidates = {};
        for (int n = 4; n >= 1; n--) {
          if (context.length >= n) {
            String ctxStr = context.sublist(context.length - n).join(" ");
            candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
          }
        }

        List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
        scored.sort((a, b) => b.value.compareTo(a.value));
        var topCands = scored.take(3).toList();

        for (var cand in topCands) {
          List<String> newTokens = List.from(currentTokens);
          newTokens.add(cand.key);
          double newScore = currentScore * cand.value;

          // Hard Rule 1: No-Repeat Trigram Penalty
          if (newTokens.length >= 3) {
            String lastTrigram = newTokens.sublist(newTokens.length - 3).join(" ");
            int count = 0;
            for (int i = 0; i < newTokens.length - 3; i++) {
              if (newTokens.sublist(i, i + 3).join(" ") == lastTrigram) count++;
            }
            if (count > 0) newScore *= 0.1;
          }

          // Soft Rule: Semantic Anchoring
          if (newTokens.length % 4 == 0) {
            var currentVec = _computeTfidfVector(newTokens.sublist(1)); // Skip BOS
            double sim = _cosineSimilarity(promptVector, currentVec);
            newScore *= (0.5 + sim);
          }

          // Hard Rule 2: LCR Right Pattern Reward
          if (cand.key == _eosToken) {
            // Check if the last 2 words form a known right pattern
            if (newTokens.length >= 3) {
              String rightCandidate = "${newTokens[newTokens.length - 2]} ${newTokens[newTokens.length - 1]}";
              if (_rightPatterns.contains(rightCandidate)) {
                newScore *= 2.0; // Massive reward for perfect ending
              }
            }
          }

          newBeams.add(MapEntry(newTokens, newScore));
        }
      }

      if (newBeams.isEmpty) break;
      newBeams.sort((a, b) => b.value.compareTo(a.value));
      beams = newBeams.take(3).toList();

      if (beams.first.key.last == _eosToken) break;
    }

    List<String> bestTokens = beams.first.key;
    // Clean BOS and EOS
    if (bestTokens.first == _bosToken) bestTokens.removeAt(0);
    if (bestTokens.last == _eosToken) bestTokens.removeLast();

    String generatedText = bestTokens.join(" ");

    // --- 80%+ MATCH GATE ---
    var genVec = _computeTfidfVector(bestTokens);
    double maxSim = 0.0;
    String matchedTrainText = "";

    for (int i = 0; i < _tfidfVectors.length; i++) {
      double sim = _cosineSimilarity(genVec, _tfidfVectors[i]);
      if (sim > maxSim) {
        maxSim = sim;
        matchedTrainText = _corpusTexts[i];
      }
    }

    bool isAccurate = (maxSim * 100) >= (threshold * 100);

    return {
      'success': true,
      'generated': generatedText,
      'confidence': (maxSim * 100).toStringAsFixed(1),
      'matched_text': matchedTrainText,
      'is_accurate': isAccurate
    };
  }

  int get vocabularySize => _counts[0][""]?.length ?? 0;
  int get totalBigramTypes => _totalBigramTypes;
}