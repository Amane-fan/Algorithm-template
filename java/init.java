import java.util.*;
import java.io.*;

public class Main {
    static FastReader fr = new FastReader();
    static PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));

    public static void main(String[] args) throws IOException {

        pw.flush();
    }

}

class FastReader {
    private final InputStream in = System.in;
    private final byte[] buffer = new byte[1 << 16];
    private int ptr = 0, len = 0;

    private int read() throws IOException {
        if (ptr >= len) {
            len = in.read(buffer);
            ptr = 0;
            if (len <= 0) return -1;
        }
        return buffer[ptr++];
    }

    int nextInt() throws IOException {
        int c;
        do {
            c = read();
        } while (c <= ' ');

        int sign = 1;
        if (c == '-') {
            sign = -1;
            c = read();
        }

        int val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = read();
        }
        return val * sign;
    }

    long nextLong() throws IOException {
        int c;
        do {
            c = read();
        } while (c <= ' ');

        int sign = 1;
        if (c == '-') {
            sign = -1;
            c = read();
        }

        long val = 0;
        while (c > ' ') {
            val = val * 10 + (c - '0');
            c = read();
        }
        return val * sign;
    }

    String next() throws IOException {
        int c;
        do {
            c = read();
        } while (c <= ' ');

        StringBuilder sb = new StringBuilder();
        while (c > ' ') {
            sb.append((char) c);
            c = read();
        }
        return sb.toString();
    }

    double nextDouble() throws IOException {
        return Double.parseDouble(next());
    }
}
