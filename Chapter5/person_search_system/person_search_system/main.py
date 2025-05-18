import sys
from PyQt5.QtWidgets import QApplication
from demo_app import DemoApp

def main():
    app = QApplication(sys.argv)
    window = DemoApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


