import sys
sys.path.append('..')
import uvicorn
from main import app

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
