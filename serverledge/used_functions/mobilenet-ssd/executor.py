from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import os
import sys
import importlib
import json
import function

hostName = "0.0.0.0"
serverPort = 8080

from io import StringIO
import sys

class CaptureOutput:
    def __enter__(self):
        self._stdout_output = ''
        self._stderr_output = ''

        self._stdout = sys.stdout
        sys.stdout = StringIO()

        self._stderr = sys.stderr
        sys.stderr = StringIO()

        return self

    def __exit__(self, *args):
        self._stdout_output = sys.stdout.getvalue()
        sys.stdout = self._stdout

        self._stderr_output = sys.stderr.getvalue()
        sys.stderr = self._stderr

    def get_stdout(self):
        return self._stdout_output

    def get_stderr(self):
        return self._stderr_output

class Executor(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode('utf-8'))

        if not "invoke" in self.path:
            self.send_response(404)
            self.end_headers()
            return

        try:
            params = request["Params"]
        except:
            params = {}

        if "context" in os.environ:
            context = json.loads(os.environ["CONTEXT"])
        else:
            context = {}

        return_output = bool(request["ReturnOutput"])

        response = {}

        try:
            if not return_output:
                result = function.handler(params, context)
                response["Output"] = ""
            else:
                with CaptureOutput() as capturer:
                    result = function.handler(params, context)
                response["Output"] = str(capturer.get_stdout()) + "\n" + str(capturer.get_stderr())

            response["Result"] = json.dumps(result)
            response["Success"] = True
        except Exception as e:
            print(e, file=sys.stderr)
            response["Success"] = False

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), "utf-8"))



if __name__ == "__main__":
    srv = HTTPServer((hostName, serverPort), Executor)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()