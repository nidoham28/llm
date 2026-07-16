import 'dart:math';
import 'dart:core';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  final List<Map<String, Map<String, int>>> _counts = List.generate(5, (_) => {});
  final List<Map<String, int>> _contextTotals = List.generate(5, (_) => {});
  final List<Map<String, int>> _distinctNext = List.generate(5, (_) => {});

  final Map<String, Set<String>> _precedingWords = {};
  int _totalBigramTypes = 0;
  static const double _D = 0.75;
  static const String _eosToken = '<EOS>'; // End of Sentence token
  final Random _random = Random();

  List<String> _tokenize(String text, {bool addEos = false}) {
    var tokens = text.toLowerCase().split(RegExp(r'\W+')).where((w) => w.isNotEmpty).toList();
    if (addEos) tokens.add(_eosToken);
    return tokens;
  }

  void train(List<Map<String, dynamic>> dbData) {
    for (int i = 0; i < 5; i++) {
      _counts[i].clear();
      _contextTotals[i].clear();
      _distinctNext[i].clear();
    }
    _precedingWords.clear();
    _totalBigramTypes = 0;

    for (var item in dbData) {
      // Add EOS token at the end of every training sentence
      var tokens = _tokenize(item['input_data'] ?? '', addEos: true);

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

  // Get top words for real-time suggestions
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
    candidates.remove(_eosToken); // Don't show EOS in live suggestions

    if (candidates.isEmpty) return [];

    List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();
    scored.sort((a, b) => b.value.compareTo(a.value));

    return scored.take(3).map((e) => e.key).toList();
  }

  // Predict with optional temperature for diversity
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
    if (candidates.isEmpty) return {'success': false, 'message': 'AI Brain needs more data.'};

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

  // Generate sentence with Temperature Sampling
  Map<String, dynamic> generateSentence(String query, int maxWords, {double temperature = 0.7}) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return {'success': false, 'message': 'Type some words to generate.'};

    List<String> generated = List.from(tokens);

    for (int i = 0; i < maxWords; i++) {
      String currentText = generated.join(" ");
      // Fetch candidates without hiding EOS
      var pred = predictNextWord(currentText, hideEos: false);
      if (!pred['success']) break;

      // Get all candidates to sample from
      List<String> context = generated.length >= 4 ? generated.sublist(generated.length - 4) : List.from(generated);
      Set<String> candidates = {};
      for (int n = 4; n >= 1; n--) {
        if (context.length >= n) {
          String ctxStr = context.sublist(context.length - n).join(" ");
          candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
        }
      }

      List<MapEntry<String, double>> scored = candidates.map((w) => MapEntry(w, _getKNProb(context, w))).toList();

      String nextWord;
      if (temperature <= 0.1) {
        // Greedy decoding (pick best)
        scored.sort((a, b) => b.value.compareTo(a.value));
        nextWord = scored.first.key;
      } else {
        // Temperature sampling
        double sum = 0;
        List<double> probs = [];
        for (var e in scored) {
          double p = pow(e.value, 1.0 / temperature).toDouble();
          probs.add(p);
          sum += p;
        }
        double rnd = _random.nextDouble() * sum;
        double cumulative = 0;
        nextWord = scored.last.key; // fallback
        for (int j = 0; j < scored.length; j++) {
          cumulative += probs[j];
          if (rnd <= cumulative) {
            nextWord = scored[j].key;
            break;
          }
        }
      }

      // Stop if AI predicts End of Sentence
      if (nextWord == _eosToken) break;

      generated.add(nextWord);
    }

    return {
      'success': true,
      'original': query,
      'generated': generated.join(" ")
    };
  }

  int get vocabularySize => _counts[0][""]?.length ?? 0;
  int get totalBigramTypes => _totalBigramTypes;
}