const express = require("express");
const userRoutes = require("./userRoutes");
const imageRoutes = require("./imageRoutes");

const router = express.Router();

router.use("/user", userRoutes);
router.use("/image", imageRoutes);

module.exports = router;
