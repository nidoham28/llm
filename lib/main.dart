import 'package:flutter/material.dart';
import 'db_helper.dart';
import 'ml_engine.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '5-Gram KN AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple, brightness: Brightness.dark),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFF0A0A12),
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _selectedIndex = 0;
  final List<Widget> _screens = const [TrainScreen(), PredictScreen()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) => setState(() => _selectedIndex = index),
        backgroundColor: Colors.black.withValues(alpha: 0.5),
        indicatorColor: Theme.of(context).colorScheme.primary.withValues(alpha: 0.3),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.model_training_outlined), selectedIcon: Icon(Icons.model_training), label: 'Train AI'),
          NavigationDestination(icon: Icon(Icons.auto_awesome_outlined), selectedIcon: Icon(Icons.auto_awesome), label: 'Predict Next'),
        ],
      ),
    );
  }
}

class GlassPanel extends StatelessWidget {
  final Widget child;
  final double padding;
  const GlassPanel({super.key, required this.child, this.padding = 20.0});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(padding),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: child,
    );
  }
}

class GradientButton extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;
  final List<Color> colors;
  const GradientButton({super.key, required this.label, required this.onPressed, this.colors = const [Colors.deepPurple, Colors.cyanAccent]});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 60,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: LinearGradient(colors: colors),
        boxShadow: [BoxShadow(color: colors.last.withValues(alpha: 0.4), blurRadius: 15, offset: const Offset(0, 4))],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onPressed,
          child: Center(child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold, letterSpacing: 1.1))),
        ),
      ),
    );
  }
}

// --- TRAIN SCREEN ---
class TrainScreen extends StatefulWidget {
  const TrainScreen({super.key});
  @override
  State<TrainScreen> createState() => _TrainScreenState();
}

class _TrainScreenState extends State<TrainScreen> {
  final TextEditingController _inputController = TextEditingController();
  final DBHelper _dbHelper = DBHelper();
  final MLEngine _mlEngine = MLEngine();

  String _configOutput = "AI Brain empty. Waiting for data...";
  bool _isTraining = false;
  List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    _refreshHistory();
  }

  void _refreshHistory() async {
    final data = await _dbHelper.getTrainingConfigs();
    _mlEngine.train(data);
    setState(() {
      _history = data;
      _configOutput = "5-Gram KN Model Ready.\nVocabulary: ${_mlEngine.vocabularySize} words\nUnique Bigram Types: ${_mlEngine.totalBigramTypes}";
    });
  }

  void _startTraining() async {
    if (_inputController.text.isEmpty) return;

    setState(() => _isTraining = true);

    Future.delayed(const Duration(milliseconds: 500), () async {
      await _dbHelper.insertTrainingConfig(_inputController.text, "Mapped");
      setState(() => _isTraining = false);
      _inputController.clear();
      _refreshHistory();
    });
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 32.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Train 5-Gram KN AI', style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.white)),
            const SizedBox(height: 8),
            Text('Feed text to build 5-gram matrices with Kneser-Ney smoothing.', style: TextStyle(color: Colors.white.withValues(alpha: 0.6), fontSize: 16)),
            const SizedBox(height: 32),

            GlassPanel(
              child: Column(
                children: [
                  TextField(
                    controller: _inputController,
                    style: const TextStyle(color: Colors.white),
                    maxLines: 3,
                    decoration: InputDecoration(
                      labelText: 'Training Sentence',
                      labelStyle: TextStyle(color: Colors.white.withValues(alpha: 0.5)),
                      border: InputBorder.none,
                      filled: true,
                      fillColor: Colors.white.withValues(alpha: 0.05),
                    ),
                  ),
                  const SizedBox(height: 20),
                  GradientButton(
                    label: _isTraining ? "SMOOTHING MATRICES..." : "TRAIN AI BRAIN",
                    onPressed: _isTraining ? () {} : _startTraining,
                    colors: const [Colors.deepPurple, Colors.purpleAccent],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),

            const Text('AI Brain Status', style: TextStyle(color: Colors.white70, fontSize: 14, letterSpacing: 1.2)),
            const SizedBox(height: 12),
            GlassPanel(
              child: _isTraining
                  ? const Center(child: CircularProgressIndicator(color: Colors.cyanAccent))
                  : Text(_configOutput, style: TextStyle(color: Colors.cyanAccent.withValues(alpha: 0.9), height: 1.5, fontFamily: 'monospace')),
            ),

            const SizedBox(height: 32),
            const Text('Trained Sentences Memory', style: TextStyle(color: Colors.white70, fontSize: 14, letterSpacing: 1.2)),
            const SizedBox(height: 12),
            _history.isEmpty
                ? Text("No data trained yet.", style: TextStyle(color: Colors.white.withValues(alpha: 0.3)))
                : Column(
              children: _history.map((item) => Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: GlassPanel(
                  padding: 12,
                  child: Row(
                    children: [
                      const Icon(Icons.memory, color: Colors.purpleAccent, size: 18),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(item['input_data'], style: TextStyle(color: Colors.white.withValues(alpha: 0.6), fontSize: 14), maxLines: 2, overflow: TextOverflow.ellipsis),
                      ),
                    ],
                  ),
                ),
              )).toList(),
            )
          ],
        ),
      ),
    );
  }
}

// --- PREDICT SCREEN ---
class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});
  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  final TextEditingController _inputController = TextEditingController();
  final DBHelper _dbHelper = DBHelper();
  final MLEngine _mlEngine = MLEngine();

  String _resultOutput = "Type a word to see AI next-word predictions...";
  bool _isPredicting = false;
  List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    _refreshHistory();
  }

  void _refreshHistory() async {
    final data = await _dbHelper.getPredictions();
    final trainData = await _dbHelper.getTrainingConfigs();
    _mlEngine.train(trainData);
    setState(() => _history = data);
  }

  void _runPrediction() async {
    if (_inputController.text.isEmpty) return;

    setState(() {
      _isPredicting = true;
      _resultOutput = "Applying Kneser-Ney Continuation Probability...";
    });

    Future.delayed(const Duration(milliseconds: 500), () async {
      var prediction = _mlEngine.predictNextWord(_inputController.text);

      String result;
      if (prediction['success']) {
        result = "Context: ${prediction['context']}\n\nPredictions:\n${prediction['predictions']}";
      } else {
        result = prediction['message'];
      }

      await _dbHelper.insertPrediction(_inputController.text, result);

      setState(() {
        _isPredicting = false;
        _resultOutput = result;
      });
      _refreshHistory();
    });
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 32.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('KN Next-Word Inference', style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.white)),
            const SizedBox(height: 8),
            Text('Type up to 4 words to trigger 5-gram predictions.', style: TextStyle(color: Colors.white.withValues(alpha: 0.6), fontSize: 16)),
            const SizedBox(height: 32),

            GlassPanel(
              child: Column(
                children: [
                  TextField(
                    controller: _inputController,
                    style: const TextStyle(color: Colors.white),
                    maxLines: 3,
                    decoration: InputDecoration(
                      labelText: 'Query (e.g. "I think that")',
                      labelStyle: TextStyle(color: Colors.white.withValues(alpha: 0.5)),
                      border: InputBorder.none,
                      filled: true,
                      fillColor: Colors.white.withValues(alpha: 0.05),
                    ),
                  ),
                  const SizedBox(height: 20),
                  GradientButton(
                    label: _isPredicting ? "PREDICTING..." : "PREDICT NEXT WORD",
                    onPressed: _isPredicting ? () {} : _runPrediction,
                    colors: const [Colors.blueGrey, Colors.cyanAccent],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),

            const Text('Prediction Engine Output', style: TextStyle(color: Colors.white70, fontSize: 14, letterSpacing: 1.2)),
            const SizedBox(height: 12),
            GlassPanel(
              child: _isPredicting
                  ? const Center(child: CircularProgressIndicator(color: Colors.cyanAccent))
                  : Text(_resultOutput, style: TextStyle(color: Colors.cyanAccent.withValues(alpha: 0.9), height: 1.8, fontFamily: 'monospace')),
            ),

            const SizedBox(height: 32),
            const Text('Prediction History', style: TextStyle(color: Colors.white70, fontSize: 14, letterSpacing: 1.2)),
            const SizedBox(height: 12),
            _history.isEmpty
                ? Text("No predictions yet.", style: TextStyle(color: Colors.white.withValues(alpha: 0.3)))
                : Column(
              children: _history.map((item) => Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: GlassPanel(
                  padding: 12,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Query: ${item['input_data']}", style: const TextStyle(color: Colors.white70, fontWeight: FontWeight.bold), maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 4),
                      Text(item['result_output'], style: TextStyle(color: Colors.cyanAccent.withValues(alpha: 0.6), fontSize: 12), maxLines: 5, overflow: TextOverflow.ellipsis),
                    ],
                  ),
                ),
              )).toList(),
            )
          ],
        ),
      ),
    );
  }
} 