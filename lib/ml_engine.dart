import 'dart:math';

class MLEngine {
  static final MLEngine _instance = MLEngine._internal();
  factory MLEngine() => _instance;
  MLEngine._internal();

  // === TRAINED SENTENCE STORAGE ===
  final List<List<String>> _sentences = [];
  final List<String> _rawTexts = [];

  // === PREFIX PATTERN MAP ===
  // "w1 w2" -> {"w3": count, "w4": count}
  final Map<String, Map<String, int>> _prefixMap = {};

  // === VOCABULARY ===
  final Set<String> _vocabulary = {};

  // === TOKENIZER ===
  // Only keeps a-z, strips everything else
  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .split(RegExp(r'[^a-z]+'))
        .where((w) => w.isNotEmpty)
        .toList();
  }

  // === WORD MATCH ===
  // Exact match OR partial prefix match ("na" matches "name")
  bool _wordMatch(String a, String b) {
    if (a == b) return true;
    if (a.length >= 2 && b.startsWith(a)) return true;
    if (b.length >= 2 && a.startsWith(b)) return true;
    return false;
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

  // === FIND ALL MATCHES ===
  // Returns top sentences sorted by match quality (exact > partial > continuation length)
  List<_Match> _findAllMatches(List<String> input, {int maxResults = 5}) {
    if (input.isEmpty || _sentences.isEmpty) return [];

    List<_Match> results = [];

    for (int s = 0; s < _sentences.length; s++) {
      var sent = _sentences[s];
      int exactLen = 0;
      int partialLen = 0;
      int limit = min(input.length, sent.length);

      for (int i = 0; i < limit; i++) {
        if (input[i] == sent[i]) {
          exactLen++;
          partialLen++;
        } else if (_wordMatch(input[i], sent[i])) {
          partialLen++;
          break; // partial stops the chain
        } else {
          break;
        }
      }

      if (partialLen > 0) {
        List<String> continuation = exactLen < sent.length
            ? sent.sublist(exactLen)
            : [];

        results.add(_Match(
          exactLen: exactLen,
          partialLen: partialLen,
          index: s,
          sentence: _rawTexts[s],
          continuation: continuation,
        ));
      }
    }

    // Sort: best partial match > best exact match > longest continuation
    results.sort((a, b) {
      if (a.partialLen != b.partialLen) {
        return b.partialLen.compareTo(a.partialLen);
      }
      if (a.exactLen != b.exactLen) {
        return b.exactLen.compareTo(a.exactLen);
      }
      return b.continuation.length.compareTo(a.continuation.length);
    });

    return results.take(maxResults).toList();
  }

  // Convenience: single best match
  _Match _findBestMatch(List<String> input) {
    var all = _findAllMatches(input, maxResults: 1);
    return all.isNotEmpty
        ? all.first
        : _Match(
      exactLen: 0,
      partialLen: 0,
      index: -1,
      sentence: '',
      continuation: [],
    );
  }

  // === GET CANDIDATES ===
  // Longest prefix first (up to 4 words), fallback to shorter
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

  // === GENERATE ONE PATH ===
  // From a match, generate one complete sentence
  List<String> _generatePath(List<String> tokens, _Match match, int maxWords) {
    List<String> result = List.from(tokens);
    Set<String> usedBigrams = {};

    if (match.continuation.isNotEmpty) {
      for (var word in match.continuation) {
        if (result.length - tokens.length >= maxWords) break;

        // Avoid repeated bigrams
        if (result.isNotEmpty) {
          String bigram = "${result.last} $word";
          if (usedBigrams.contains(bigram)) break;
          usedBigrams.add(bigram);
        }
        result.add(word);
      }
    } else {
      for (int step = 0; step < maxWords; step++) {
        var cands = _getCandidates(result);
        if (cands.isEmpty) break;

        String chosen;
        var sent = _sentences[match.index];
        if (match.index >= 0 && result.length < sent.length) {
          chosen = sent[result.length];
        } else {
          chosen = cands.first.key;
        }

        // Skip if same as last word
        if (result.isNotEmpty && result.last == chosen) {
          if (cands.length > 1) {
            chosen = cands[1].key;
          } else {
            break;
          }
        }

        // Skip if bigram already used
        String bigram = "${result.last} $chosen";
        if (usedBigrams.contains(bigram)) {
          bool foundAlt = false;
          for (var c in cands) {
            String altBigram = "${result.last} ${c.key}";
            if (!usedBigrams.contains(altBigram) && c.key != result.last) {
              chosen = c.key;
              foundAlt = true;
              break;
            }
          }
          if (!foundAlt) break;
        }

        usedBigrams.add("${result.last} $chosen");
        result.add(chosen);
      }
    }

    return result;
  }

  // === PUBLIC API ===

  int get vocabularySize => _vocabulary.length;
  int get totalBigramTypes => _prefixMap.length;

  // Live suggestions — also handles partial last word
  List<String> getLiveSuggestions(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) return [];

    var cands = _getCandidates(tokens);

    // If nothing found, the last token might be a partial word
    // Try getting candidates without it, then filter by prefix
    if (cands.isEmpty && tokens.length > 1 && tokens.last.length >= 2) {
      String partial = tokens.last;
      cands = _getCandidates(tokens.sublist(0, tokens.length - 1));
      cands = cands.where((e) => e.key.startsWith(partial)).toList();
    }

    return cands.take(3).map((e) => e.key).toList();
  }

  // Predict next word with pattern match %
  Map<String, dynamic> predictNextWord(String query) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) {
      return {'success': false, 'message': 'Type some words.'};
    }

    var match = _findBestMatch(tokens);
    double matchPct =
    tokens.isNotEmpty ? (match.partialLen / tokens.length) * 100 : 0;

    var cands = _getCandidates(tokens);

    // Partial last word fallback
    if (cands.isEmpty && tokens.length > 1 && tokens.last.length >= 2) {
      String partial = tokens.last;
      cands = _getCandidates(tokens.sublist(0, tokens.length - 1));
      cands = cands.where((e) => e.key.startsWith(partial)).toList();
    }

    if (cands.isEmpty) {
      return {
        'success': false,
        'message': 'No patterns found. Train more sentences.',
      };
    }

    var top3 = cands.take(3).toList();
    int total = top3.fold(0, (sum, e) => sum + e.value);

    // If partial match on last word, boost candidates that complete it
    String matchType = match.partialLen == match.exactLen ? '' : ' (partial)';

    return {
      'success': true,
      'context':
      "Pattern Match: ${matchPct.toStringAsFixed(0)}%$matchType | \"${match.sentence}\"",
      'predictions': top3
          .map((e) =>
      "  -> ${e.key}  [${((e.value / total) * 100).toStringAsFixed(1)}%]")
          .join('\n'),
      'best_word': top3.first.key,
    };
  }

  // Generate sentence — returns up to 3 options with match %
  Map<String, dynamic> generateSentence(String query, int maxWords,
      {double threshold = 0.80}) {
    var tokens = _tokenize(query);
    if (tokens.isEmpty) {
      return {'success': false, 'message': 'Type some words to generate.'};
    }

    // Find all matching sentences (up to 3)
    var matches = _findAllMatches(tokens, maxResults: 3);

    // Generate one path per match
    List<String> options = [];
    List<double> pcts = [];
    String bestPattern = '';

    for (var match in matches) {
      List<String> result = _generatePath(tokens, match, maxWords);

      double inputPct =
      tokens.isNotEmpty ? (match.partialLen / tokens.length) * 100 : 0;

      String text = result.join(' ');

      // Skip duplicate options
      if (options.contains(text)) continue;

      options.add(text);
      pcts.add(inputPct);
      if (bestPattern.isEmpty) bestPattern = match.sentence;
    }

    // Fallback: no matches at all, generate from prefix map
    if (options.isEmpty) {
      List<String> result = List.from(tokens);
      Set<String> usedBigrams = {};

      for (int step = 0; step < maxWords; step++) {
        var cands = _getCandidates(result);
        if (cands.isEmpty) break;

        String chosen = cands.first.key;
        if (result.isNotEmpty && result.last == chosen) {
          if (cands.length > 1) {
            chosen = cands[1].key;
          } else {
            break;
          }
        }

        String bigram = "${result.last} $chosen";
        if (usedBigrams.contains(bigram)) {
          bool foundAlt = false;
          for (var c in cands) {
            if (!usedBigrams.contains("${result.last} ${c.key}") &&
                c.key != result.last) {
              chosen = c.key;
              foundAlt = true;
              break;
            }
          }
          if (!foundAlt) break;
        }

        usedBigrams.add("${result.last} $chosen");
        result.add(chosen);
      }

      options.add(result.join(' '));
      pcts.add(0);
    }

    // Best option is always first
    String best = options.first;
    double bestPct = pcts.first;

    // Output match: how much of the full generated text matches a trained sentence
    var bestTokens = best.split(' ');
    var outputMatch = _findBestMatch(bestTokens);
    double outputPct =
    bestTokens.isNotEmpty ? (outputMatch.partialLen / bestTokens.length) * 100 : 0;

    return {
      'success': true,
      'generated': best,
      'confidence': bestPct.toStringAsFixed(1),
      'output_match': outputPct.toStringAsFixed(1),
      'matched_text': bestPattern,
      'is_accurate': bestPct >= (threshold * 100),
      'options': options,
      'options_match': pcts.map((p) => p.toStringAsFixed(0)).toList(),
    };
  }
}

// Helper: pattern match result
class _Match {
  final int exactLen; // Fully matching leading words
  final int partialLen; // Exact + partial prefix matches
  final int index; // Sentence index (-1 if none)
  final String sentence; // Raw text
  final List<String> continuation; // Words after the match point

  _Match({
    required this.exactLen,
    required this.partialLen,
    required this.index,
    required this.sentence,
    required this.continuation,
  });
}