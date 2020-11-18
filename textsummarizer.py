import ssl
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class SummarizerServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        print("%s %s" % (self.command, self.path))
        print("headers = [%s]" % self.headers)
        self._set_headers()
        self.wfile.write('{"statusCode":"0"}')

    def do_POST(self):
        print("%s: %s " % (self.command, self.path))
        print("headers = [%s]" % self.headers)
        self._set_headers()
        if self.path == "/summarize":
            length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(length)
            data = json.loads(post_data)
            summary = self.getSummary(data["transcript"])
            self.wfile.write(bytes(summary,encoding='utf8'))
        else:
            self.wfile.write('{"statusCode":"0"}')

    def getSummary(self,text):
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')

        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        print ("original text preprocessed: \n", preprocess_text)

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


        # summmarize
        summary_ids = model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        print ("\n\nSummarized text: \n",output)

        return output


def run(server_class=HTTPServer, handler_class=SummarizerServer, port=8089):
    # server_address = ('', port)
    # httpd = server_class(server_address, handler_class)
    # print ('Starting httpd on port %d...'%port)
    # httpd.serve_forever()

    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    ssl_context = ssl.SSLContext(
        ssl.PROTOCOL_SSLv23)  # @UndefinedVariable
    # if a wrong password is provided this function can freeze
    ssl_context.load_cert_chain(u'server.pem', password='genesysscreenrecording')
    httpd.socket = ssl_context.wrap_socket(httpd.socket,
                                   server_side=True)
    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
