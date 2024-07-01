const express = require("express");
const imageController = require("../controllers/imageController");

const router = express.Router();

router.post("/upload", imageController.uploadImage);
router.get("/:category", imageController.getImagesByCategory);
router.get("/", imageController.getCountByUsercode);

module.exports = router;
