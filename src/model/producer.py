class Producer:

    def __init__(self):
        self.real_summary      = []
        self.predicted_summary = []
        pass

    def update_summary(self, real_spans, predicted_spans):
        self.real_summary     .extend(real_spans)
        self.predicted_summary.extend(predicted_spans)
        pass

    def show_summary(self):

        with open('real_summary.txt', 'w') as filehandle:   
            print("Expected summary:")
            for span in self.real_summary:
                filehandle.write('%s\n' % span)
                print(span)
                print()

        with open('predicted_summary.txt', 'w') as filehandle:   
            print("Produced summary:")
            for span in self.predicted_summary:
                filehandle.write('%s\n' % span)
                print(span)
                print()
        pass