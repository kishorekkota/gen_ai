import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:html' as html;

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Doc Verifier',
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: const UploadPage(),
    );
  }
}

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  String? prediction;
  String? validity;
  List<String> metadata = [];

  void uploadFile(html.File file) {
    final request = html.HttpRequest();
    final formData = html.FormData();
    formData.appendBlob('file', file, file.name);

    request.open('POST', 'http://localhost:8000/upload/');
    request.onLoadEnd.listen((event) {
      final doc = html.DocumentFragment.html(request.responseText!);
      final label = doc.querySelector("h2")?.text ?? '';
      final status = doc.querySelector("h3")?.text ?? '';
      final metaList = doc.querySelectorAll("ul li").map((e) => e.text!).toList();

      setState(() {
        prediction = label;
        validity = status;
        metadata = metaList;
      });
    });

    request.send(formData);
  }

  void pickFile() {
    html.FileUploadInputElement input = html.FileUploadInputElement();
    input.accept = ".pdf,.png,.jpg";
    input.click();

    input.onChange.listen((event) {
      final file = input.files?.first;
      if (file != null) uploadFile(file);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Doc Verifier")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          ElevatedButton(onPressed: pickFile, child: const Text("Upload Document")),
          if (prediction != null) ...[
            const SizedBox(height: 20),
            Text("Prediction: $prediction", style: const TextStyle(fontSize: 18)),
            Text("Status: $validity", style: const TextStyle(fontSize: 18)),
            const Text("Metadata:", style: TextStyle(fontWeight: FontWeight.bold)),
            for (var m in metadata) Text(m),
          ]
        ]),
      ),
    );
  }
}
