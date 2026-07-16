import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

class DBHelper {
  static final DBHelper _instance = DBHelper._internal();
  factory DBHelper() => _instance;
  DBHelper._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB();
    return _database!;
  }

  Future<Database> _initDB() async {
    String path = await getDatabasesPath();
    return openDatabase(
      join(path, 'hybrid_llm_v2.db'), // New DB name to avoid schema conflicts
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE training_configs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_data TEXT,
            config_output TEXT,
            timestamp TEXT
          )
        ''');
        await db.execute('''
          CREATE TABLE predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_data TEXT,
            result_output TEXT,
            timestamp TEXT
          )
        ''');
      },
      version: 1,
    );
  }

  Future<int> insertTrainingConfig(String inputData, String configOutput) async {
    final db = await database;
    return await db.insert('training_configs', {
      'input_data': inputData,
      'config_output': configOutput,
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  Future<List<Map<String, dynamic>>> getTrainingConfigs() async {
    final db = await database;
    return await db.query('training_configs', orderBy: 'timestamp DESC');
  }

  Future<int> insertPrediction(String inputData, String resultOutput) async {
    final db = await database;
    return await db.insert('predictions', {
      'input_data': inputData,
      'result_output': resultOutput,
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  Future<List<Map<String, dynamic>>> getPredictions() async {
    final db = await database;
    return await db.query('predictions', orderBy: 'timestamp DESC');
  }
}