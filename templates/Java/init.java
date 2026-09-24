import java.nio.charset.StandardCharsets;
import java.util.*;
import java.io.*;

public class Main {
    static FastReader fr = new FastReader();
    static PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out));
    static StringBuilder sb = new StringBuilder();

    public static void main(String[] args) throws IOException {

        pw.flush();
    }

}

class FastReader {
    private final InputStream in;
    private final byte[] buffer = new byte[1 << 16];
    private int ptr = 0, len = 0;

    private final ByteArrayOutputStream text =
            new ByteArrayOutputStream(128);

    FastReader() {
        this(System.in);
    }

    FastReader(InputStream in) {
        this.in = in;
    }

    private int read() throws IOException {
        if (ptr >= len) {
            len = in.read(buffer);
            ptr = 0;
            if (len <= 0) return -1;
        }
        return buffer[ptr++] & 0xff;
    }

    private void unread(int c) {
        if (c != -1) ptr--;
    }

    private int skipWhitespace() throws IOException {
        int c;
        do {
            c = read();
        } while (c != -1 && c <= ' ');
        return c;
    }

    int nextInt() throws IOException {
        int c = skipWhitespace();
        if (c == -1) throw new EOFException("没有更多整数");

        boolean negative = c == '-';
        if (c == '-' || c == '+') c = read();

        int val = 0;
        while (c >= '0' && c <= '9') {
            val = val * 10 - (c - '0');
            c = read();
        }
        unread(c);

        return negative ? val : -val;
    }

    long nextLong() throws IOException {
        int c = skipWhitespace();
        if (c == -1) throw new EOFException("没有更多整数");

        boolean negative = c == '-';
        if (c == '-' || c == '+') c = read();

        long val = 0;
        while (c >= '0' && c <= '9') {
            val = val * 10 - (c - '0');
            c = read();
        }
        unread(c);

        return negative ? val : -val;
    }

    String next() throws IOException {
        int c = skipWhitespace();
        if (c == -1) return null;

        text.reset();
        while (c != -1 && c > ' ') {
            text.write(c);
            c = read();
        }
        unread(c);

        return text.toString(StandardCharsets.UTF_8.name());
    }

    String nextLine() throws IOException {
        int c = read();
        if (c == -1) return null;

        text.reset();
        while (c != -1 && c != '\n' && c != '\r') {
            text.write(c);
            c = read();
        }

        if (c == '\r') {
            int next = read();
            if (next != '\n') unread(next);
        }

        return text.toString(StandardCharsets.UTF_8.name());
    }

    double nextDouble() throws IOException {
        String s = next();
        if (s == null) throw new EOFException("没有更多浮点数");
        return Double.parseDouble(s);
    }
}
