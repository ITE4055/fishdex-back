const express = require("express");
const mysql = require("mysql");
const bodyParser = require("body-parser");
const jwt = require("jsonwebtoken");

const app = express();
const port = 3000;

// MySQL 연결 설정
const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "1415",
  database: "test",
});

// MySQL 연결
db.connect((err) => {
  if (err) {
    throw err;
  }
  console.log("MySQL Connected");
});

// Body parser middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

// 로그인 엔드포인트
app.post("/login", (req, res) => {
  const { username, usercode } = req.body;
  if (!username || !usercode) {
    return res
      .status(400)
      .json({ message: "사용자명과 회원번호를 입력해주세요." });
  }

  // MySQL에서 사용자 이름 중복 확인
  const checkQuery = `SELECT * FROM users WHERE usercode = ?`;
  db.query(checkQuery, [usercode], (err, results) => {
    if (err) {
      console.log(err);
      return res.status(500).json({ message: "Internal server error" });
    }
    if (results.length > 0) {
      // return res.status(409).json({ message: "이미 등록된 회원입니다." });
      const token = jwt.sign({ usercode }, "your_secret_key", {
        expiresIn: "1h",
      });
      return res.status(200).json({ message: "로그인 완료", token });
    }

    // 새로운 사용자 추가
    const insertQuery = `INSERT INTO users (username, usercode) VALUES (?, ?)`;
    db.query(insertQuery, [username, usercode], (err, results) => {
      if (err) {
        console.log(err);
        return res.status(500).json({ message: "Internal server error" });
      }
      // 사용자 추가 후 JWT 발급
      const token = jwt.sign({ usercode }, "your_secret_key", {
        expiresIn: "1h",
      });
      return res.status(200).json({ message: "회원 등록 완료", token });
    });
  });
});

app.post("/upload", (req, res) => {
  const { usercode, url, category } = req.body;

  const insertQuery = `INSERT INTO images (usercode, url, category) VALUES (?, ?, ?)`;
  // usercode 유효성 판별, 카테고리 유효성 판별이 필요함

  db.query(insertQuery, [usercode, url, category], (err, results) => {
    if (err) {
      console.log(err);
      return res.status(500).json({ message: "Internal server error" });
    }
    console.log("이미지 등록 완료");
    return res.status(200).json({ message: "이미지 등록 완료" });
  });
});

app.get("/download/:category", (req, res) => {
  const { usercode } = req.body;
  const category = req.params.category;

  const query = `SELECT url FROM images WHERE usercode = ? AND category = ?`;
  db.query(query, [usercode, category], (err, results, fields) => {
    if (err) {
      return res.status(500).json({ message: "Internal server error" });
    }
    // 결과값이 없으면
    if (results.length === 0) {
      return res
        .status(401)
        .json({ message: "결과값이 없습니다. 도감을 채워주세요." });
    }
    const urls = results.map((result) => result.url);
    return res.status(200).json(urls);
  });
});

// 서버 시작
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
