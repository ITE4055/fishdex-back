const express = require("express");
const routes = require("./src/routes");

const app = express();

// 미들웨어 설정
app.use(express.json());

// 라우트 설정
app.use("/api", routes);

module.exports = app;
