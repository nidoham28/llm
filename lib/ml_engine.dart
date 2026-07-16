import 'dart:math';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  // --- N-Gram Storage (1 to 5-grams) ---
  // _counts[level][context][nextWord] = count
  final List<Map<String, Map<String, int>>> _counts = List.generate(5, (_) => {});
  final List<Map<String, int>> _contextTotals = List.generate(5, (_) => {});
  final List<Map<String, int>> _distinctNext = List.generate(5, (_) => {});

  // For Kneser-Ney Continuation Probability (Unigram base case)
  final Map<String, Set<String>> _precedingWords = {}; // Tracks unique previous words for each word
  int _totalBigramTypes = 0;

  static const double _D = 0.75; // Standard Discount value for Kneser-Ney

  List<String> _tokenize(String text) {
    return text.toLowerCase().split(RegExp(r'\W+')).where((w) => w.isNotEmpty).toList();
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

    for (var item in dbData) {
      var tokens = _tokenize(item['input_data'] ?? '');

      for (int i = 0; i < tokens.length; i++) {
        // 1-gram (Context is empty "")
        _updateCount(0, "", tokens[i]);

        // Track continuation probability for unigrams
        if (i > 0) {
          _precedingWords.putIfAbsent(tokens[i], () => {});
          if (_precedingWords[tokens[i]]!.add(tokens[i - 1])) {
            _totalBigramTypes++; // Increment unique bigram types
          }
        }

        // 2-gram to 5-gram
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

    bool isNewWord = !_counts[level][context]!.containsKey(word);
    if (isNewWord) {
      _distinctNext[level][context] = _distinctNext[level][context]! + 1;
    }

    _counts[level][context]![word] = (_counts[level][context]![word] ?? 0) + 1;
    _contextTotals[level][context] = _contextTotals[level][context]! + 1;
  }

  // Recursive Interpolated Kneser-Ney Probability
  double _getKNProb(List<String> context, String word) {
    if (context.isEmpty) {
      // Base Case: Continuation Probability
      int distinctPreceding = _precedingWords[word]?.length ?? 0;
      if (_totalBigramTypes == 0) return 0.0;
      return distinctPreceding / _totalBigramTypes;
    }

    int level = context.length - 1;
    String ctxStr = context.join(" ");

    int count = _counts[level][ctxStr]?[word] ?? 0;
    int total = _contextTotals[level][ctxStr] ?? 0;
    int distinctNext = _distinctNext[level][ctxStr] ?? 0;

    if (total == 0) {
      // Backoff if context never seen
      return _getKNProb(context.sublist(1), word);
    }

    // Kneser-Ney Formula
    double firstTerm = max(count - _D, 0) / total;
    double lambda = (_D / total) * distinctNext;
    double continuationProb = _getKNProb(context.sublist(1), word);

    return firstTerm + (lambda * continuationProb);
  }

  Map<String, dynamic> predictNextWord(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) {
      return {'success': false, 'message': 'Type some words to predict the next one.'};
    }

    // Extract up to 4 words for 5-gram prediction
    List<String> context = tokens.length >= 4
        ? tokens.sublist(tokens.length - 4)
        : List.from(tokens);

    // Gather candidate words from all overlapping contexts
    Set<String> candidates = {};

    // Add words from 5-gram down to 2-gram contexts
    for (int n = 4; n >= 1; n--) {
      if (context.length >= n) {
        String ctxStr = context.sublist(context.length - n).join(" ");
        candidates.addAll(_counts[n]?[ctxStr]?.keys ?? []);
      }
    }

    // Add top general continuation words if we need more options
    if (candidates.length < 10) {
      var topCont = _precedingWords.entries.toList()
        ..sort((a, b) => b.value.length.compareTo(a.value.length));
      for (var e in topCont.take(20)) {
        candidates.add(e.key);
      }
    }

    if (candidates.isEmpty) {
      return {'success': false, 'message': 'AI Brain needs more data to make predictions.'};
    }

    // Calculate KN Probability for all candidates
    List<MapEntry<String, double>> scoredCandidates = [];
    for (var word in candidates) {
      double prob = _getKNProb(context, word);
      scoredCandidates.add(MapEntry(word, prob));
    }

    // Sort by highest probability
    scoredCandidates.sort((a, b) => b.value.compareTo(a.value));
    var top3 = scoredCandidates.take(3).toList();

    // Format Output
    String contextStr = context.join(" ");
    int nGramLevel = min(5, context.length + 1);

    String predictionsStr = top3.map((e) {
      double percentage = e.value * 100;
      return "  ↳ ${e.key}  [KN Prob: ${percentage.toStringAsFixed(2)}%]";
    }).join('\n');

    return {
      'success': true,
      'context': "[$nGramLevel-Gram Context] '$contextStr'",
      'predictions': predictionsStr,
      'best_word': top3.first.key
    };
  }

  int get vocabularySize => _counts[0][""]?.length ?? 0;
  int get totalBigramTypes => _totalBigramTypes;
}