import 'dart:math';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  // === TRAINED SENTENCE STORAGE ===
  final List<List<String>> _sentences = [];
  final List<String> _rawTexts = [];

  // === PREFIX PATTERN MAP ===
  // "w1 w2" -> {"w3": count, "w4": count, ...}
  final Map<String, Map<String, int>> _prefixMap = {};

  // === VOCABULARY ===
  final Set<String> _vocabulary = {};

  // === TOKENIZER ===
  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .split(RegExp(r'[^a-z]+'))
        .where((w) => w.isNotEmpty)
        .toList();
  }

  // === TRAIN ===
  void train(List<Map<String, dynamic>> dbData) {
    _sentences.clear();
    _rawTexts.clear();
    _prefixMap.clear();
    _vocabulary.clear();

    for (var item in dbData) {
      String raw = (item['input_data'] ?? '').trim();
      if (raw.isEmpty) continue;

      for (var sentence in raw.split(RegExp(r'[.!?\n]+'))) {
        sentence = sentence.trim();
        if (sentence.isEmpty) continue;

        var words = _tokenize(sentence);
        if (words.length < 2) continue;

        _sentences.add(words);
        _rawTexts.add(sentence);

        for (int i = 0; i < words.length; i++) {
          _vocabulary.add(words[i]);

          for (int prefixLen = 1; prefixLen <= min(i, 4); prefixLen++) {
            String prefix = words.sublist(i - prefixLen, i).join(' ');
            _prefixMap.putIfAbsent(prefix, () => {});
            _prefixMap[prefix]![words[i]] =
                (_prefixMap[prefix]![words[i]] ?? 0) + 1;
          }
        }
      }
    }
  }

  // === PATTERN MATCH ===
  _Match _findBestMatch(List<String> input) {
    if (input.isEmpty || _sentences.isEmpty) {
      return _Match(0, -1, '', []);
    }

    int bestLen = 0;
    int bestIdx = -1;

    for (int s = 0; s < _sentences.length; s++) {
      var sent = _sentences[s];
      int len = 0;
      int limit = min(input.length, sent.length);

      while (len < limit && input[len] == sent[len]) {
        len++;
      }

      if (len > bestLen ||
          (len == bestLen && len > 0 && sent.length > _sentences[bestIdx].length)) {
        bestLen = len;
        bestIdx = s;
        if (bestLen == input.length) break;
      }
    }

    if (bestIdx < 0) return _Match(0, -1, '', []);

    List<String> continuation = bestLen < _sentences[bestIdx].length
        ? _sentences[bestIdx].sublist(bestLen)
        : [];

    return _Match(bestLen, bestIdx, _rawTexts[bestIdx], continuation);
  }

  // === GET CANDIDATES ===
  List<MapEntry<String, int>> _getCandidates(List<String> context) {
    if (context.isEmpty) return [];

    for (int len = min(context.length, 4); len >= 1; len--) {
      String prefix = context.sublist(context.length - len).join(' ');
      var next = _prefixMap[prefix];
      if (next != null && next.isNotEmpty) {
        return next.entries.toList()
          ..sort((a, b) => b.value.compareTo(a.value));
      }
    }
    return [];
  }

  // === PUBLIC API ===

  int get vocabularySize => _vocabulary.length;
  int get totalBigramTypes => _prefixMap.length;

  List<String> getLiveSuggestions(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return [];

    var cands = _getCandidates(tokens);
    return cands.take(3).map((e) => e.key).toList();
  }

  Map<String, dynamic> predictNextWord(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return {'success': false, 'message': 'Type some words.'};

    var match = _findBestMatch(tokens);
    double matchPct =
    tokens.isNotEmpty ? (match.length / tokens.length) * 100 : 0;

    var cands = _getCandidates(tokens);
    if (cands.isEmpty) {
      return {
        'success': false,
        'message': 'No patterns found. Train more sentences.'
      };
    }

    var top3 = cands.take(3).toList();
    int total = top3.fold(0, (sum, e) => sum + e.value);

    return {
      'success': true,
      'context':
      "Pattern Match: ${matchPct.toStringAsFixed(0)}% | \"${match.sentence}\"",
      'predictions': top3
          .map((e) =>
      "  -> ${e.key}  [${((e.value / total) * 100).toStringAsFixed(1)}%]")
          .join('\n'),
      'best_word': top3.first.key,
    };
  }

  Map<String, dynamic> generateSentence(String query, int maxWords,
      {double threshold = 0.80}) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) {
      return {'success': false, 'message': 'Type some words to generate.'};
    }

    List<String> result = List.from(tokens);

    var match = _findBestMatch(tokens);
    double inputMatchPct =
    tokens.isNotEmpty ? (match.length / tokens.length) * 100 : 0;

    if (match.index >= 0 && match.continuation.isNotEmpty) {
      for (var word in match.continuation) {
        if (result.length - tokens.length >= maxWords) break;
        result.add(word);
      }
    } else if (match.index >= 0) {
      for (int step = 0; step < maxWords; step++) {
        var cands = _getCandidates(result);
        if (cands.isEmpty) break;

        var sent = _sentences[match.index];
        String chosen;

        if (result.length < sent.length) {
          chosen = sent[result.length];
        } else {
          chosen = cands.first.key;
        }

        if (result.isNotEmpty && result.last == chosen) break;
        result.add(chosen);
      }
    } else {
      for (int step = 0; step < maxWords; step++) {
        var cands = _getCandidates(result);
        if (cands.isEmpty) break;

        String chosen = cands.first.key;

        if (result.isNotEmpty && result.last == chosen) break;
        result.add(chosen);
      }
    }

    String generatedText = result.join(' ');

    var finalMatch = _findBestMatch(result);
    double outputMatchPct =
    result.isNotEmpty ? (finalMatch.length / result.length) * 100 : 0;

    return {
      'success': true,
      'generated': generatedText,
      'confidence': inputMatchPct.toStringAsFixed(1),
      'output_match': outputMatchPct.toStringAsFixed(1),
      'matched_text': match.sentence,
      'is_accurate': inputMatchPct >= (threshold * 100),
    };
  }
}

class _Match {
  final int length;
  final int index;
  final String sentence;
  final List<String> continuation;

  _Match(this.length, this.index, this.sentence, this.continuation);
}