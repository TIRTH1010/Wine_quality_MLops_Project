import argparse


if __name__=='__main__':
    try:
        args=argparse.ArgumentParser()
        args.add_argument("--name","-n",default='okok',type=str)
        args.add_argument("--age","-a",default=25.0,type=float)
        parsing=args.parse_args()


        print(parsing.name,parsing.age)
    except Exception as e:
        print(e)