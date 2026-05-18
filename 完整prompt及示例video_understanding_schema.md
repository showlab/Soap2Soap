# 视频理解 Prompt Schema 文档

本文档展示了 `video_to_script2.py` 中所采用的视频理解系统的完整 Prompt Schema，以及实际运行后生成的镜头描述示例。

---

## 目录

1. [系统架构概览](#系统架构概览)
2. [Prompt Schema 详解](#prompt-schema-详解)
   - [阶段1：全局角色识别](#阶段1全局角色识别)
   - [阶段2：大场景检测](#阶段2大场景检测)
   - [阶段3：场景服装档案](#阶段3场景服装档案)
   - [阶段4：镜头分析](#阶段4镜头分析)
3. [输出数据结构](#输出数据结构)
4. [镜头描述示例](#镜头描述示例)

---

## 系统架构概览

该视频理解系统采用多阶段分析流程：

```
视频输入
    ↓
[场景检测] → 自动切分镜头
    ↓
[阶段1] 全局角色识别 → 建立角色档案
    ↓
[阶段2] 大场景检测 → 识别环境/地点变化
    ↓
[阶段3] 场景服装档案 → 记录每个大场景中的角色服装
    ↓
[阶段4] 镜头分析 → 对每个镜头进行深度电影化分析
    ↓
输出 JSON 文件
```

---

## Prompt Schema 详解

### 阶段1：全局角色识别

**目的**：分析整个视频，识别所有独特角色并建立档案。

**Prompt 结构**：

```
You are analyzing an entire video to identify ALL unique characters that appear throughout.

--- AUDIO TRANSCRIPT (FULL VIDEO) ---
{完整的语音转录文本}

--- YOUR TASK ---
Watch the entire video and create a comprehensive character roster.

对于每个角色，需要提取：

1. **角色ID分配**
   - 格式：@character_XX（XX为01, 02, 03等）
   - 基于首次出场顺序

2. **角色名称提取**（最高优先级）
   A. 从屏幕文字标签提取（首次出场时）
      - 主名称（primary_name）
      - 头衔（titles）
      - 角色（roles）

   B. 名称分类
      - primary_name: 主要/全名（如 "Emma Smith"）
      - aliases: 昵称、简称（如 "Em", "Emmy"）
      - titles: 正式头衔（如 "Dr. Chen", "Professor"）
      - roles: 故事角色（如 "Protagonist", "Doctor"）
      - familial: 家庭称呼（如 "Mom", "Dad"）

   C. 来源追踪
      - "on_screen_label": 视频中的文字标签
      - "dialogue": 对话中提到
      - "id_fallback": 未找到名称

3. **识别特征描述**
   - 物理特征：性别、年龄、体型、肤色
   - 头发：颜色、长度、发型
   - 面部：胡须、眼镜、明显标记
   - 服装：不同场景中的服装变化
   - 配饰：帽子、首饰、包、眼镜
   - 独特特征：区分于他人的特质

4. **出场信息**
   - 首次出场时间
   - 出现场景列表
```

**输出示例**：
```json
{
  "characters": [
    {
      "character_id": "@character_01",
      "names": {
        "primary_name": "Emma Smith",
        "primary_source": "on_screen_label",
        "aliases": ["Emma", "Em"],
        "aliases_sources": {"Emma": "on_screen_label", "Em": "dialogue"},
        "titles": ["Ms. Smith"],
        "titles_sources": {"Ms. Smith": "dialogue"}
      },
      "physical_attributes": "Female, young adult, slim build, fair skin",
      "hair": "Red, curly, long",
      "face": "Fair complexion, red lips",
      "clothing_variations": [
        {"scene": "Scene 1", "description": "Light blue business suit"}
      ],
      "first_appearance": "0.0s",
      "scenes": ["Scene 1", "Scene 3", "Scene 5"]
    }
  ]
}
```

---

### 阶段2：大场景检测

**目的**：识别视频中的大场景变化（环境/地点变化）。

**Prompt 结构**：

```
You are analyzing a video to identify MAJOR SCENES (locations/environments).

--- YOUR TASK ---
Watch the entire video and identify all major scenes where the environment/setting changes.

大场景定义：
- 地点变化（如：办公室 → 家 → 餐厅）
- 光照变化（如：日光 → 室内人工光）
- 场景变化（不同的背景环境）

不是大场景：
- 同一房间内的摄像机角度变化
- 同一地点的不同景别
- 轻微的摄像机移动

对于每个大场景，需要提供：

1. **基本信息**
   - scene_id: 唯一标识（major_scene_01, major_scene_02等）
   - start_time / end_time: 起止时间
   - duration: 持续时间
   - location_type: 地点类型描述
   - setting_description: 场景简述
   - lighting_style: 主导光照风格
   - color_palette: 色彩调色板

2. **环境描述**（environment_description）
   【关键：只描述环境，不包含人物】

   包含内容：
   ✅ 房间布局和建筑结构
   ✅ 墙壁（颜色、材质、纹理）
   ✅ 地板（类型、颜色、图案、纹理）
   ✅ 窗户（大小、位置、样式、窗外景观）
   ✅ 门（类型、位置、把手样式）
   ✅ 家具（类型、位置、颜色、材质、形状）
   ✅ 照明设备（类型、位置、色温、强度）
   ✅ 装饰元素（艺术品、植物、地毯、窗帘、物品）
   ✅ 氛围（光照质量、情绪、一天中的时间）

   排除内容：
   ❌ 所有人物、角色、人形
   ❌ 角色动作或移动
   ❌ 角色服装或面孔
   ❌ 对话或语音
```

**输出示例**：
```json
{
  "major_scenes": [
    {
      "scene_id": "major_scene_01",
      "start_time": 0.0,
      "end_time": 65.5,
      "duration": 65.5,
      "location_type": "Luxury penthouse living room",
      "setting_description": "Modern minimalist room with floor-to-ceiling windows",
      "lighting_style": "Natural daylight from windows, cool ambient fill",
      "color_palette": "Cool greys, whites, blues",
      "environment_description": "A modern minimalist living room with floor-to-ceiling windows on the back wall offering a city view..."
    }
  ]
}
```

---

### 阶段3：场景服装档案

**目的**：为每个大场景建立角色服装档案，确保同一场景内服装描述一致。

**Prompt 结构**：

```
You are analyzing a specific time segment of a video to create EXACT clothing descriptions for characters.

--- TIME SEGMENT ---
Scene ID: {场景ID}
Start: {起始时间}s
End: {结束时间}s

--- GLOBAL CHARACTER ROSTER ---
{全局角色档案}

--- YOUR TASK ---
Watch this time segment carefully and document what each character is wearing in THIS SCENE.

【服装DNA提取系统】（7维度详细规格）

1. **颜色系统**（必须使用精确颜色识别）
   - Primary Color: Pantone TCX代码 + HEX值 + 通用名称
   - Secondary Colors: 相同格式
   - Pattern Colors: 相同格式
   - 格式: "Pantone 18-0303 TCX (#8B8C8E) - Warm Grey"

2. **面料系统**
   - Material: 类型（棉、丝、羊毛、皮革、亚麻、合成混纺）
   - Weave: 平纹、斜纹、缎纹、针织、梭织
   - Weight: g/m² 估算（轻 <150g，中 150-250g，重 >250g）
   - Opacity: 不透明 / 半透明 / 透明
   - Finish: 哑光 / 缎面（光泽） / 金属 / 纹理
   - Stretch: 无 / 2向 / 4向
   - Texture: 光滑 / 粗糙 / 天鹅绒 / 颗粒状
   - Drape: 挺括 / 垂坠 / 僵硬

3. **剪裁与版型系统**
   上装：
   - Fit: 修身 / 常规 / 宽松 / 超大
   - Length: 短款 / 腰部 / 臀部 / 大腿 / 膝盖 / 全长
   - Shoulder: 自然 / 落肩 / 插肩 / 装袖
   - Sleeve: 无袖 / 短袖 / 3/4袖 / 长袖
   - Collar: V领 / 圆领 / 有领 / 翻领 / 无领
   - Closure: 纽扣 / 拉链 / 套头 / 系带

   下装：
   - Fit: 紧身 / 直筒 / 宽松 / 喇叭 / 休闲
   - Length: 短裤 / 七分裤 / 九分裤 / 长裤
   - Waist: 高腰 / 中腰 / 低腰
   - Hem: 直筒 / 锥形 / 喇叭 / 翻边
   - Pockets: 类型和位置

4. **细节系统**
   - Buttons: 数量、位置、颜色（Pantone+HEX）、材质、大小
   - Zipper: 位置、颜色、类型（隐形/金属）、长度
   - Pockets: 类型（贴袋/斜插袋/嵌线袋）、位置、翻盖
   - Stitching: 线色、类型（明线/包缝）、间距
   - 独特特征：褶皱、袖口、内衬等

5. **图案系统**（如适用）
   - Type: 几何 / 花卉 / 抽象 / 民族 / 纯色
   - Size: 小 / 中 / 大
   - Arrangement: 规则 / 不规则
   - Direction: 水平 / 垂直 / 对角
   - Density: 稀疏 / 中等 / 密集
   - Colors: 列出所有图案颜色（Pantone+HEX）

6. **配饰系统**
   - Shoes: 类型、颜色（Pantone+HEX）、材质、跟高
   - Jewelry: 项链、耳环、手镯、戒指、手表
   - Bags: 类型、颜色、大小、材质
   - 其他可见配饰

7. **造型系统**
   - Layering: 单层 / 双层 / 多层
   - Tuck: 塞入 / 半塞 / 不塞
   - Jacket state: 敞开 / 扣上（扣几颗）
   - Sleeve state: 卷起 / 未卷
   - 整体风格描述词
```

**输出示例**：
```json
{
  "scene_id": "major_scene_01",
  "character_wardrobe": {
    "@character_01": {
      "top": {
        "item_name": "Edwardian Lace Tea Dress Bodice",
        "color": {
          "primary": {
            "pantone_tc": "Pantone 11-0701 TCX",
            "hex": "#F0EAD6",
            "name": "Whisper White"
          }
        },
        "fabric": {
          "material": "Lace over Silk",
          "weave": "Intricate lace knit",
          "weight": "Light (<150g)",
          "opacity": "Semi-transparent overlay"
        },
        "cut": {
          "fit": "Tailored/Fitted",
          "sleeve": "Long",
          "collar": "High-neck collar"
        }
      },
      "full_description": "Rose is dressed in a refined Edwardian tea-length gown..."
    }
  }
}
```

---

### 阶段4：镜头分析

**目的**：对每个镜头进行深度电影化分析。

**Prompt 结构**：

```
You will analyze a video {segment_type} from {start_time}s to {end_time}s (duration: {duration}s).

--- TIME INFORMATION ---
Scene Start: {起始时间}s
Scene End: {结束时间}s
Scene Duration: {持续时间}s

--- MAJOR SCENE CONTEXT ---
This shot belongs to: {所属大场景ID}

--- CHARACTER NAMING GUIDE ---
{角色命名指南}

--- SCENE WARDROBE ---
{场景服装档案}

--- GLOBAL CHARACTER ROSTER ---
{全局角色档案}

--- AUDIO TRANSCRIPT ---
{当前镜头的字幕转录}

【任务1：技术电影化提取】

1. **光照与色彩**
   - lighting_setup: 光源类型、方向、硬度
     例如："Harsh sunlight", "Rim light", "Soft window light"
   - color_grading: 色彩倾向、对比度、LUT风格
     例如："Cool blues", "Teal and Orange", "Desaturated"

2. **构图与氛围**
   - composition: 元素排列
     例如："Rule of thirds", "Center symmetry", "Leading lines"
   - mood_atmosphere: 抽象感觉和心理暗示
     例如："Tense", "Epic", "Melancholic"

3. **摄像机几何（3D空间）**
   - shot_size: 主体在画面中的大小
     例如："Wide Shot", "Medium Close-up"
   - camera_angle: 垂直角度
     例如："Low Angle", "High Angle", "Eye-level"
   - camera_height: 离地面的物理高度
     例如："Waist-Level", "Ground-Level"
   - horizontal_angle: 相对于主体的角度
     例如："Frontal", "Three-Quarter", "Profile"

4. **技术规格（光学质感）**
   - focal_length: 透视感
     例如："80mm telephoto", "24mm wide"
   - depth_of_field: 背景虚化
     例如："f/1.8 Shallow focus", "f/8 Deep focus"
   - tech_device: 摄像机/镜头元数据
     例如："IMAX MSM 9802", "Kodak Vision3 500T", "Anamorphic lens"

5. **运动动态**
   - camera_movement: 摄像机如何移动
     例如："Static", "Tracking shot", "Handheld shake"
   - subject_movement: 特定角色或环境如何移动
     必须使用严格ID，并引用转录中的语音/动作
     格式："@character_01 turns head and speaks: [2.5s] 你好世界"

【任务2：叙事 I2V Prompt】

写一个单一、高度详细的电影化段落，描述此时间段：
- 必须整合所有提取的技术细节成连贯的叙事流
- 涵盖场景、环境、光照、详细角色动作、空间动态
- **关键**：包含转录中相关的对话/语音内容
- **重要**：I2V Prompt中不包含时间戳注释（如[2.5s]），自然使用对话
- **关键**：使用严格ID（@character_01, @character_02）引用所有角色
- 将角色所说的内容（来自转录）与其在此片段中的视觉外观和动作匹配
- 无项目符号。一个连续的散文块

【关键构图规则】：
- **单一镜头**：只描述一个连续的镜头/帧
- **无分屏构图**：不描述分屏、双联画、三联画、网格布局
- **无文字/图形**：不包含文字叠加、字幕、标题、水印
- **无序列布局**：只描述一个时间点，不是多个时刻并排显示
- **纯粹电影场景**：专注于电影场景本身

【任务3：Language to One Shot Reference Prompt】

基于I2V Prompt创建一个静态关键帧参考描述。

目的：
从I2V Prompt中提取并保留视觉元素，创建静态关键帧参考。
此关键帧将作为主参考，用于保持{major_scene_id}中所有镜头的视觉一致性。

保留内容（静态元素）：
- 环境/场景描述
- 光照设置
- 摄像机几何（景别、角度、高度、构图）
- 角色位置和姿势（静态，非移动）
- 面部表情
- 服装和外观细节
- 构图和取景
- 情绪和氛围
- 色彩分级和视觉风格

移除内容（动态元素）：
- ❌ 对话/引用
- ❌ 动作动词（走、跑、转、动、手势等）
- ❌ 移动描述（穿过、朝向、远离等）
- ❌ 时间序列指示词（然后、接着、突然等）
- ❌ 过渡短语
- ❌ 时间戳

转换示例：
I2V Prompt（动态）："Emma walks across the room towards the window, turns her head, and asks 'What do you think?' with a curious expression."

Keyframe（静态）："Emma stands in the middle of the room, her body oriented towards a large window on the back wall. She wears a light blue business suit and has a curious, inquiring expression on her face..."

输出格式：
写一个单一、全面的段落描述静态关键帧
- 使用现在时
- 专注于视觉元素
- 无动作动词
- 无对话
- 无移动
- 纯粹的冻结时刻视觉描述
```

**输出格式**：
```json
{
  "lighting_setup": "String description...",
  "color_grading": "String description...",
  "composition": "String description...",
  "mood_atmosphere": "String description...",
  "shot_size": "String description...",
  "camera_angle": "String description...",
  "camera_height": "String description...",
  "horizontal_angle": "String description...",
  "focal_length": "String description...",
  "depth_of_field": "String description...",
  "tech_device": "String description...",
  "camera_movement": "String description...",
  "subject_movement": "String description with IDs and dialogue timestamps",
  "I2V Prompt": "A single, long, deeply detailed cinematic paragraph...",
  "Language_to_One_Shot_Prompt": "A comprehensive STATIC KEYFRAME reference description..."
}
```

---

## 输出数据结构

最终输出的JSON文件包含以下顶级结构：

```json
{
  "video_file": "视频文件路径",
  "video_metadata": {
    "aspect_ratio": {
      "width": 1920,
      "height": 1080,
      "aspect_ratio": "16:9",
      "ratio_decimal": 1.777778
    }
  },
  "total_scenes": 20,
  "character_roster": {
    "characters": [ /* 角色档案数组 */ ]
  },
  "major_scenes": {
    "major_scenes": [ /* 大场景数组 */ ]
  },
  "scene_wardrobe": {
    "scene_wardrobes": {
      "major_scene_01": { /* 场景1的服装档案 */ },
      "major_scene_02": { /* 场景2的服装档案 */ }
    }
  },
  "scenes": [ /* 镜头分析数组 */ ]
}
```

---

## 镜头描述示例

以下是从 `disney_titannictest.json` 中提取的一个完整镜头描述示例：

### 镜头 8：首次相遇

**基本信息**：
- **时间范围**：24.67s - 32.03s
- **持续时间**：7.37秒
- **所属大场景**：major_scene_01（Ship Steerage General Room）

---

#### 技术电影化参数

| 参数 | 描述 |
|------|------|
| **光照设置** | 柔和、漫射的日光从上层甲板楼梯间射入，创造光晕效果，在统舱舱位中扩散为更柔和的环境光，用温柔的光辉照亮主角的面部。 |
| **色彩分级** | 自然主义的时代色调，统舱环境的泥土棕和灰（Pantone Kangaroo、Peat）与闯入者裙装的明亮、纯净的Whisper White和Cornsilk Yellow形成对比。 |
| **构图** | 视平线中等景别，利用正反打结构；最初框定长椅上的一组人，具有深度层次，过渡到站立人物在白色柱子之间的中心、框中框构图。 |
| **氛围情绪** | 悬停的时间感和阶级对比的时刻；惊讶、好奇，以及在日常环境中的一丝浪漫张力。 |
| **景别** | Medium Shot transitioning to Medium Close-up |
| **摄像机角度** | Eye-level |
| **摄像机高度** | Seated level to Standing level |
| **水平角度** | Frontal and Three-Quarter views |
| **焦距** | 50mm to 85mm standard cinematic lenses |
| **景深** | Shallow depth of field (f/2.8) isolating faces against the blurred, busy background |
| **技术设备** | 35mm Film Camera (Panavision Panaflex), Anamorphic Lenses |
| **摄像机运动** | Static framing with minimal movement |

---

#### 主体运动描述

```
@character_03 leans forward and points, speaking: [24.9s] 'Jack.'
@character_02 turns his head to look.
@character_01 stands still and speaks: [30.0s] 'Hello, Mr. Dawson.'
```

---

#### I2V Prompt（动态叙事段落）

> In the crowded, hazy atmosphere of the steerage common room, @character_03 leans forward eagerly on a wooden bench, pointing off-screen to alert his friend, saying 'Jack.' Seated next to him, @character_02, dressed in his Kangaroo brown corduroy shirt and suspenders, turns his head with a look of genuine surprise. The perspective reveals @character_01 standing elegantly amidst the rougher surroundings, her Whisper White lace dress and vibrant Citrus yellow sash glowing in the soft diffused light. She maintains a poised, upper-class demeanor as she looks directly at him and speaks, 'Hello, Mr. Dawson.' @character_02 stares back, stunned by her appearance in this unexpected place.

---

#### Language to One Shot Prompt（静态关键帧参考）

> @character_02 and @character_03 sit on a wooden bench in the steerage common room, bathed in soft, hazy daylight. @character_02 wears a Kangaroo grey-brown corduroy shirt with beige suspenders and looks towards the camera with a curtain-parted hairstyle. Beside him, @character_03 wears a Copper Coin tan wool vest, a white collarless shirt with rolled sleeves, and a dark newsboy cap, pointing excitedly with his right hand. The background is filled with other passengers in muted earth-tone clothing, blurring into the bright, diffused light from the windows. The scene captures a detailed period texture with a naturalistic, slightly warm color palette.

---

### 该镜头的角色服装详情

#### @character_02 (Jack) 的服装

**上装：Corduroy Work Shirt**
- **颜色**：Pantone 18-0617 TCX (#776A5F) - Kangaroo（柔和的灰棕色）
- **面料**：棉质灯芯绒，中厚（150-250g），哑光/天鹅绒质感，垂直肋状纹理
- **剪裁**：常规版型，尖领，4孔纽扣，两个带翻盖的胸前贴袋
- **完整描述**：Jack穿着一件实用的Kangaroo灰棕色灯芯绒衬衫，内搭可见领口的白色Henley打底衫，衬衫塞入厚重的Peat深炭色长裤，由带有细垂直条纹的米色背带固定。

---

### 视觉理解要点

这个镜头展示了系统如何：

1. **角色一致性**：正确识别并使用 `@character_01`、`@character_02`、`@character_03` 等ID
2. **对话整合**：将转录中的对话（如 "Jack." 和 "Hello, Mr. Dawson."）与视觉描述融合
3. **服装精确性**：使用Pantone色彩系统和详细的面料描述
4. **电影语言**：准确描述景别、焦距、景深等技术参数
5. **静态/动态分离**：I2V Prompt包含动作和对话，而Language to One Shot Prompt冻结为一个静态时刻

---

## 总结

该视频理解系统的核心优势：

1. **多阶段处理**：从全局到局部，逐步细化理解
2. **角色追踪**：在整个视频中保持角色ID一致性
3. **服装档案**：为大场景中的每个角色建立详细的服装DNA
4. **电影化分析**：专业的镜头语言描述
5. **双重输出**：动态I2V Prompt用于视频生成，静态Keyframe用于一致性参考

---

*文档生成时间：2025年*
*基于 video_to_script2.py 和 disney_titannictest.json*
