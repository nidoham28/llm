import 'dart:math';
import 'dart:core';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  // --- N-Gram & Skip-gram Storage ---
  final List<Map<String, Map<String, int>>> _counts = List.generate(5, (_) => {});
  final List<Map<String, int>> _contextTotals = List.generate(5, (_) => {});
  final List<Map<String, int>> _distinctNext = List.generate(5, (_) => {});

  // Skip-gram storage (Context with gaps)
  final Map<String, Map<String, int>> _skipgramMap = {};

  // --- TF-IDF Storage (For Fallback & Semantic Anchor) ---
  final List<String> _corpusTexts = [];
  final List<Map<String, double>> _tfidfVectors = [];
  Map<String, double> _idf = {};
  Set<String> _vocabulary = {};

  final Map<String, Set<String>> _precedingWords = {};
  int _totalBigramTypes = 0;
  static const double _D = 0.75;
  static const String _eosToken = '<EOS>';
  final Random _random = Random();

  // Simple Subword/Root extractor (Pseudo-BPE)
  String _getRoot(String word) {
    if (word.endsWith('ing')) return word.substring(0, word.length - 3);
    if (word.endsWith('s')) return word.substring(0, word.length - 1);
    if (word.endsWith('ed')) return word.substring(0, word.length - 2);
    return word;
  }

  List<String> _tokenize(String text, {bool addEos = false}) {
    var tokens = text.toLowerCase().split(RegExp(r'\W+')).where((w) => w.isNotEmpty).toList();
    if (addEos) tokens.add(_eosToken);
    return tokens;
  }

  void train(List<Map<String, dynamic>> dbData) {
    // Clear memory
    for (int i = 0; i < 5; i++) {
      _counts[i].clear();
      _contextTotals[i].clear();
      _distinctNext[i].clear();
    }
    _skipgramMap.clear();
    _precedingWords.clear();
    _totalBigramTypes = 0;

    // TF-IDF Clear
    _corpusTexts.clear();
    _tfidfVectors.clear();
    _idf.clear();
    _vocabulary.clear();

    List<List<String>> tokenizedDocs = [];
    Map<String, int> docFrequency = {};

    for (var item in dbData) {
      var tokens = _tokenize(item['input_data'] ?? '', addEos: true);
      _corpusTexts.add(item['input_data'] ?? '');
      tokenizedDocs.add(tokens);

      // TF-IDF Vocab building
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

        // 2-gram to 5-gram
        for (int n = 2; n <= 5; n++) {
          if (i >= n - 1) {
            String context = tokens.sublist(i - n + 1, i).join(" ");
            _updateCount(n - 1, context, tokens[i]);
          }
        }

        // Skip-grams (e.g., word1 + word3 -> word4)
        if (i >= 2) {
          String skipContext = "${tokens[i - 2]} _ ${tokens[i - 1]}";
          _skipgramMap.putIfAbsent(skipContext, () => {});
          if (i + 1 < tokens.length) {
            _skipgramMap[skipContext]![tokens[i + 1]] = (_skipgramMap[skipContext]![tokens[i + 1]] ?? 0) + 1;
          }
        }
      }
    }

    // Calculate TF-IDF for fallback
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
      String root = _getRoot(token); // Apply Pseudo-BPE
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

    int level = context.length - 1;
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

        // Add Skip-gram candidates
        if (n >= 2) {
          String skipCtx = "${context[context.length - n]} _ ${context[context.length - 1]}";
          candidates.addAll(_skipgramMap[skipCtx]?.keys ?? []);
        }
      }
    }
    candidates.remove(_eosToken);

    if (candidates.isEmpty) return [];

    List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
    scored.sort((a, b) => b.value.compareTo(a.value));

    return scored.take(3).map((e) => e.key).toList();
  }

  Map<String, dynamic> predictNextWord(String query, {bool hideEos = true}) {
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
    if (hideEos) candidates.remove(_eosToken);

    if (candidates.isEmpty) {
      return {'success': false, 'message': 'No patterns found. Try training more data.'};
    }

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

  // --- DEEP THINK: BEAM SEARCH GENERATION ---
  Map<String, dynamic> generateSentence(String query, int maxWords, {double temperature = 0.7}) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return {'success': false, 'message': 'Type some words to generate.'};

    // Semantic anchor: TF-IDF of the prompt
    var promptVector = _computeTfidfVector(tokens);

    // Beam Search state: List of possible sentences (tokens + score)
    List<MapEntry<List<String>, double>> beams = [
      MapEntry(List.from(tokens), 1.0) // Initial beam
    ];

    for (int step = 0; step < maxWords; step++) {
      List<MapEntry<List<String>, double>> newBeams = [];

      for (var beam in beams) {
        List<String> currentTokens = beam.key;
        double currentScore = beam.value;

        var pred = predictNextWord(currentTokens.join(" "), hideEos: false);
        if (!pred['success']) continue;

        List<String> context = currentTokens.length >= 4 ? currentTokens.sublist(currentTokens.length - 4) : List.from(currentTokens);
        Set<String> candidates = {};
        for (int n = 4; n >= 1; n--) {
          if (context.length >= n) {
            String ctxStr = context.sublist(context.length - n).join(" ");
            candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
          }
        }

        // Expand top 3 candidates for each beam
        List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
        scored.sort((a, b) => b.value.compareTo(a.value));
        var topCands = scored.take(3).toList();

        for (var cand in topCands) {
          List<String> newTokens = List.from(currentTokens);
          newTokens.add(cand.key);

          double newScore = currentScore * cand.value; // Multiply probabilities

          // 1. No-Repeat Trigram Penalty (Prevents AI loops like "I think I think")
          if (newTokens.length >= 3) {
            String lastTrigram = newTokens.sublist(newTokens.length - 3).join(" ");
            int count = 0;
            for (int i = 0; i < newTokens.length - 3; i++) {
              if (newTokens.sublist(i, i + 3).join(" ") == lastTrigram) count++;
            }
            if (count > 0) newScore *= 0.1; // Heavy penalty for repetition
          }

          // 2. Semantic Anchoring (Keeps sentence relevant to prompt)
          if (newTokens.length % 4 == 0) { // Check every 4 words
            var currentVec = _computeTfidfVector(newTokens);
            double sim = _cosineSimilarity(promptVector, currentVec);
            newScore *= (0.5 + sim); // Reward for staying on topic
          }

          // 3. Reward EOS (Encourages finishing the sentence naturally)
          if (cand.key == _eosToken) {
            newScore *= 1.5;
          }

          newBeams.add(MapEntry(newTokens, newScore));
        }
      }

      if (newBeams.isEmpty) break;

      // Sort new beams by score and keep top 3
      newBeams.sort((a, b) => b.value.compareTo(a.value));
      beams = newBeams.take(3).toList();

      // If the best beam ended with EOS, we can stop early
      if (beams.first.key.last == _eosToken) {
        beams.first.key.removeLast(); // Remove EOS token from final text
        break;
      }
    }

    // Clean up EOS from final output if present
    List<String> bestTokens = beams.first.key;
    if (bestTokens.last == _eosToken) bestTokens.removeLast();

    return {
      'success': true,
      'original': query,
      'generated': bestTokens.join(" ")
    };
  }

  int get vocabularySize => _counts[0][""]?.length ?? 0;
  int get totalBigramTypes => _totalBigramTypes;
}