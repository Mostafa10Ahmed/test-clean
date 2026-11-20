import fs from 'fs';
import {
  Client,
  GatewayIntentBits,
  REST,
  Routes,
  SlashCommandBuilder,
  EmbedBuilder,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  AttachmentBuilder,
  ModalBuilder,
  TextInputBuilder,
  TextInputStyle,
  PermissionFlagsBits,
} from 'discord.js';
import { ChapterCleaningManager } from './chapterProcessor';
import { BubbleCleanMode } from '../engine/types';
import { cleanWithRemoteWorker, isRemoteWorkerConfigured } from './remoteWorkerClient';
import {
  createAuthUrl,
  completeManualCodeFromUser,
  downloadFromDriveLink,
  getDriveLink,
  getFallbackTokenJson,
  parseDriveUrl,
  resolveDriveAuth,
  unlinkDrive,
  updateDriveFolder,
  uploadToDrive,
} from './driveOAuth';

export interface DiscordGatewayStatus {
  configured: boolean;
  connected: boolean;
  botTag?: string;
  registeredCommands: string[];
}

const DISCORD_MAX_ATTACHMENT_BYTES = 24 * 1024 * 1024; // Safe ceiling under 25MB

export class BearDiscordBot {
  private client: Client | null = null;
  private manager: ChapterCleaningManager;
  private connected = false;
  private botTag?: string;

  public readonly slashCommands = [
    '/clean',
    '/profile',
    '/detect',
    '/review',
    '/status',
    '/cancel',
    '/settings',
    '/export',
    '/help',
  ];

  constructor(manager: ChapterCleaningManager) {
    this.manager = manager;
  }

  public getStatus(): DiscordGatewayStatus {
    return {
      configured: Boolean(process.env.DISCORD_BOT_TOKEN || process.env.DISCORD_TOKEN),
      connected: this.connected,
      botTag: this.botTag,
      registeredCommands: this.slashCommands,
    };
  }

  private buildProfilePayload(userId: string, guildId: string | null, isAdmin: boolean) {
    const userLink = getDriveLink('user', userId);
    const guildLink = guildId ? getDriveLink('guild', guildId) : null;
    const fallback = getFallbackTokenJson();

    const embed = new EmbedBuilder()
      .setTitle('👤 بروفايلك — ربط Google Drive')
      .setDescription(
        'اربط حساب Google Drive بتاعك عشان البوت يستقبل روابط الدرايف (فولدر أو ZIP أو صورة) ويرفعلك الفصول المتبيّضة مباشرة على مساحتك.'
      )
      .addFields(
        {
          name: '📂 درايفك الشخصي',
          value: userLink
            ? `✅ مربوط: **${userLink.email}**\n[فتح فولدر الرفع](${userLink.folderUrl})`
            : '❌ غير مربوط',
          inline: false,
        },
        {
          name: '🏠 درايف السيرفر',
          value: guildLink
            ? `✅ مربوط: **${guildLink.email}**\n[فتح فولدر السيرفر](${guildLink.folderUrl})`
            : '❌ غير مربوط',
          inline: false,
        },
        {
          name: '🤖 الدرايف الاحتياطي (token.json)',
          value: fallback ? `✅ جاهز كـ fallback (${fallback.email})` : '— غير متوفر',
          inline: false,
        }
      )
      .setColor(0x0f172a);

    const row1 = new ActionRowBuilder<ButtonBuilder>().addComponents(
      new ButtonBuilder()
        .setCustomId('profile:link_user')
        .setLabel('🔗 ربط درايفك (Google)')
        .setStyle(ButtonStyle.Primary),
      new ButtonBuilder()
        .setCustomId('profile:paste_code:user')
        .setLabel('📋 إدخال كود الربط')
        .setStyle(ButtonStyle.Secondary),
      new ButtonBuilder()
        .setCustomId('profile:set_folder:user')
        .setLabel('📁 تحديد فولدر الرفع')
        .setStyle(ButtonStyle.Secondary),
      new ButtonBuilder()
        .setCustomId('profile:unlink_user')
        .setLabel('🗑️ فك الربط')
        .setStyle(ButtonStyle.Danger)
        .setDisabled(!userLink)
    );

    const row2 = new ActionRowBuilder<ButtonBuilder>().addComponents(
      new ButtonBuilder()
        .setCustomId('profile:refresh')
        .setLabel('🔄 تحديث')
        .setStyle(ButtonStyle.Success)
    );

    if (guildId && isAdmin) {
      row2.addComponents(
        new ButtonBuilder()
          .setCustomId('profile:link_guild')
          .setLabel('🏠 ربط درايف السيرفر')
          .setStyle(ButtonStyle.Primary),
        new ButtonBuilder()
          .setCustomId('profile:paste_code:guild')
          .setLabel('📋 كود السيرفر')
          .setStyle(ButtonStyle.Secondary),
        new ButtonBuilder()
          .setCustomId('profile:unlink_guild')
          .setLabel('🗑️ فك درايف السيرفر')
          .setStyle(ButtonStyle.Danger)
          .setDisabled(!guildLink)
      );
    }

    return { embeds: [embed], components: [row1, row2] };
  }

  public async startIfConfigured(): Promise<void> {
    const token = process.env.DISCORD_BOT_TOKEN || process.env.DISCORD_TOKEN;
    if (!token || token.trim() === '') {
      return;
    }

    try {
      this.client = new Client({
        intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent],
      });

      this.client.once('ready', async (readyClient) => {
        this.connected = true;
        this.botTag = readyClient.user.tag;
        console.log(`Discord bot connected as ${this.botTag}`);
        const guildIds = Array.from(readyClient.guilds.cache.keys());
        await this.registerSlashCommands(token, readyClient.user.id, guildIds);
      });

      // Prefix command: $sync (or @Bot sync) to clear old commands & instantly refresh slash commands
      this.client.on('messageCreate', async (message) => {
        if (message.author.bot || !message.guild) return;
        const raw = message.content.trim();
        const botId = this.client?.user?.id;
        const mentionMatch = botId && new RegExp(`^<@!?${botId}>\\s+sync(\\s+clear)?$`, 'i').test(raw);
        const prefixMatch = /^\$sync(\s+clear)?$/i.test(raw);
        if (!prefixMatch && !mentionMatch) return;

        const adminRoleId = (process.env.ADMIN_ROLE_ID || '').trim();
        const hasAdminPerm = Boolean(
          message.member?.permissions.has(PermissionFlagsBits.Administrator) ||
            message.member?.permissions.has(PermissionFlagsBits.ManageGuild) ||
            (adminRoleId && message.member?.roles.cache.has(adminRoleId))
        );
        if (!hasAdminPerm) {
          await message.reply('❌ الأمر `$sync` للأدمن بس.').catch(() => {});
          return;
        }

        const statusMsg = await message.reply('🔄 جاري مسح الأوامر القديمة وتحديث أوامر السلاش في السيرفر...').catch(() => null);
        try {
          if (!botId) throw new Error('Bot client ID unavailable');
          const rest = new REST({ version: '10' }).setToken(token);
          // Clear old guild commands first, then push fresh commands to this guild & globally
          await rest.put(Routes.applicationGuildCommands(botId, message.guild.id), { body: [] });
          await this.registerSlashCommands(token, botId, [message.guild.id]);
          const cmdList = this.slashCommands.map((c) => `\`${c}\``).join(' · ');
          const okText = `✅ تم مسح الأوامر القديمة وتحديث أوامر السلاش فوراً في السيرفر!\nالأوامر المتاحة: ${cmdList}`;
          if (statusMsg) {
            await statusMsg.edit(okText).catch(() => {});
          } else {
            await message.reply(okText).catch(() => {});
          }
        } catch (err) {
          const errText = `❌ فشل تحديث الأوامر: ${(err as Error).message}`;
          if (statusMsg) {
            await statusMsg.edit(errText).catch(() => {});
          } else {
            await message.reply(errText).catch(() => {});
          }
        }
      });

      this.client.on('interactionCreate', async (interaction) => {
        try {
          // --- MODAL SUBMISSIONS (/profile manual code or custom Drive folder) ---
          if (interaction.isModalSubmit()) {
            if (interaction.customId.startsWith('modal:drive_code:')) {
              await interaction.deferReply({ ephemeral: true });
              const scope = interaction.customId.endsWith(':guild') ? 'guild' : 'user';
              const ownerId = scope === 'guild' ? interaction.guildId || interaction.user.id : interaction.user.id;
              const rawInput = interaction.fields.getTextInputValue('code_input');
              const rec = await completeManualCodeFromUser(scope, ownerId, rawInput);
              await interaction.editReply({
                content: `✅ تم ربط Google Drive بنجاح (**${rec.email}**)\n📂 فولدر الرفع: ${rec.folderUrl}`,
              });
              return;
            }
            if (interaction.customId.startsWith('modal:drive_folder:')) {
              await interaction.deferReply({ ephemeral: true });
              const scope = interaction.customId.endsWith(':guild') ? 'guild' : 'user';
              const ownerId = scope === 'guild' ? interaction.guildId || interaction.user.id : interaction.user.id;
              const folderInput = interaction.fields.getTextInputValue('folder_input');
              const updated = updateDriveFolder(scope, ownerId, folderInput);
              if (!updated) {
                await interaction.editReply({
                  content: '❌ اربط حسابك بجوجل الأول من زر **🔗 ربط درايفك** وبعدين حدّد الفولدر.',
                });
                return;
              }
              await interaction.editReply({
                content: `✅ تم تحديث فولدر الرفع: ${updated.folderUrl}`,
              });
              return;
            }
          }

          // --- SLASH COMMANDS ---
          if (interaction.isChatInputCommand()) {
            const cmdName = `/${interaction.commandName}`;

            // /profile — Interactive Google Drive linking panel
            if (interaction.commandName === 'profile') {
              const isAdmin = Boolean(
                interaction.memberPermissions?.has(PermissionFlagsBits.ManageGuild) ||
                  interaction.memberPermissions?.has(PermissionFlagsBits.Administrator)
              );
              const payload = this.buildProfilePayload(interaction.user.id, interaction.guildId, isAdmin);
              await interaction.reply({ ...payload, ephemeral: true });
              return;
            }

            await interaction.deferReply();

            // Dedicated /clean handler with explicit clean_sfx + file upload OR Google Drive link + Drive upload
            if (interaction.commandName === 'clean') {
              const cleanSfx = interaction.options.getBoolean('clean_sfx', true);
              const attachment = interaction.options.getAttachment('file');
              const driveLinkOpt = interaction.options.getString('drive_link')?.trim() || '';
              const uploadToDriveOpt = interaction.options.getBoolean('upload_to_drive');
              const outputFolderOpt = interaction.options.getString('output_drive_folder')?.trim() || '';
              const chapterOpt = interaction.options.getString('chapter') || '';
              const bubbleModeOpt = interaction.options.getString('bubble_mode');
              const bubbleCleanMode: BubbleCleanMode | undefined =
                bubbleModeOpt === 'text_only'
                  ? 'TEXT_ONLY'
                  : bubbleModeOpt === 'whiten'
                    ? 'WHITEN'
                    : undefined;

              if (attachment || driveLinkOpt) {
                let inputBuffer: Buffer;
                let fileName: string;
                let driveAuth: Awaited<ReturnType<typeof resolveDriveAuth>> = null;
                let sourceFolderId: string | undefined;

                if (driveLinkOpt) {
                  driveAuth = await resolveDriveAuth(interaction.user.id, interaction.guildId);
                  if (!driveAuth) {
                    await interaction.editReply({
                      content: '📂 مفيش حساب Google Drive مربوط. اربط حسابك الأول من `/profile` وبعدين جرّب تاني.',
                    });
                    return;
                  }
                  const downloaded = await downloadFromDriveLink(driveLinkOpt, driveAuth.accessToken);
                  inputBuffer = downloaded.buffer;
                  fileName = downloaded.fileName;
                  sourceFolderId = downloaded.sourceFolderId;
                } else {
                  const response = await fetch(attachment!.url);
                  if (!response.ok) {
                    throw new Error(`Failed to download attached file (${response.status})`);
                  }
                  inputBuffer = Buffer.from(await response.arrayBuffer());
                  fileName = attachment!.name || 'page.png';
                }

                let outputPath = '';
                let outputFileName = '';
                let totalPages = 1;
                let cleanedRegions = 0;
                let totalProcessingTimeMs = 0;
                let finalBubbleMode: BubbleCleanMode = bubbleCleanMode || 'WHITEN';
                let jobId = '';
                let cleanupOutputPath = false;

                if (isRemoteWorkerConfigured()) {
                  const remote = await cleanWithRemoteWorker({
                    fileName,
                    buffer: inputBuffer,
                    cleanSfx,
                    bubbleCleanMode,
                  });
                  outputPath = remote.outputPath;
                  outputFileName = remote.outputFileName;
                  totalPages = remote.totalPages;
                  cleanedRegions = remote.cleanedRegions;
                  totalProcessingTimeMs = remote.totalProcessingTimeMs;
                  finalBubbleMode = remote.bubbleCleanMode;
                  jobId = remote.jobId;
                  cleanupOutputPath = true;
                } else {
                  const job = await this.manager.ingestUploadedFiles(
                    [{ originalname: fileName, buffer: inputBuffer }],
                    `Discord: ${fileName}`,
                    { enableSFX: cleanSfx, bubbleCleanMode }
                  );
                  totalPages = job.totalPages;
                  cleanedRegions = job.cleanedRegions;
                  totalProcessingTimeMs = job.totalProcessingTimeMs;
                  finalBubbleMode = job.settings.bubbleCleanMode;
                  jobId = job.id;
                  if (job.pages.length === 1 && job.pages[0].cleanedUrl) {
                    outputPath = this.manager.resolveDiskPathFromUrl(job.pages[0].cleanedUrl);
                    outputFileName = `${job.pages[0].fileName.replace(/\.[^.]+$/, '')}_cleaned.png`;
                  } else if (job.zipDownloadUrl) {
                    outputPath = this.manager.resolveDiskPathFromUrl(job.zipDownloadUrl);
                    outputFileName = `${job.id}_cleaned.zip`;
                  }
                }

                try {
                  const fileSize = outputPath && fs.existsSync(outputPath) ? fs.statSync(outputPath).size : 0;
                  const shouldUploadToDrive =
                    uploadToDriveOpt === true ||
                    (uploadToDriveOpt !== false && (Boolean(driveLinkOpt) || fileSize > DISCORD_MAX_ATTACHMENT_BYTES));

                  let uploadedDriveLink = '';
                  let driveSourceLabel = '';

                  if (shouldUploadToDrive && outputPath && fs.existsSync(outputPath)) {
                    if (!driveAuth) {
                      driveAuth = await resolveDriveAuth(interaction.user.id, interaction.guildId);
                    }
                    if (driveAuth) {
                      let targetFolderId = driveAuth.folderId;
                      if (outputFolderOpt) {
                        const parsedTarget = parseDriveUrl(outputFolderOpt);
                        if (parsedTarget?.id) targetFolderId = parsedTarget.id;
                      } else if (sourceFolderId && driveLinkOpt) {
                        targetFolderId = driveAuth.folderId;
                      }
                      const outBuf = await fs.promises.readFile(outputPath);
                      const mimeType = outputFileName.toLowerCase().endsWith('.png')
                        ? 'image/png'
                        : 'application/zip';
                      const up = await uploadToDrive({
                        accessToken: driveAuth.accessToken,
                        parentFolderId: targetFolderId,
                        fileName: outputFileName,
                        buffer: outBuf,
                        mimeType,
                      });
                      uploadedDriveLink = up.webViewLink;
                      driveSourceLabel = driveAuth.sourceLabel;
                    }
                  }

                  const filesToSend: AttachmentBuilder[] = [];
                  if (
                    outputPath &&
                    fs.existsSync(outputPath) &&
                    fileSize <= DISCORD_MAX_ATTACHMENT_BYTES &&
                    !uploadedDriveLink
                  ) {
                    filesToSend.push(new AttachmentBuilder(outputPath, { name: outputFileName }));
                  }

                  const embed = new EmbedBuilder()
                    .setTitle(`Cleaned: ${fileName}`)
                    .setDescription(
                      `Processed **${totalPages}** page(s) · **${cleanedRegions}** regions cleaned in **${(totalProcessingTimeMs / 1000).toFixed(2)}s**${isRemoteWorkerConfigured() ? ' (Remote Worker)' : ''}.` +
                        (uploadedDriveLink
                          ? `\n\n📂 **[فتح النتيجة على Google Drive](${uploadedDriveLink})** (${driveSourceLabel})`
                          : '')
                    )
                    .addFields(
                      {
                        name: 'SFX Cleaning',
                        value: cleanSfx ? 'Enabled (SFX Cleaned)' : 'Disabled (SFX Untouched)',
                        inline: true,
                      },
                      {
                        name: 'Bubble Mode',
                        value:
                          finalBubbleMode === 'TEXT_ONLY'
                            ? 'Text-Only Removal'
                            : 'Smart Whitening (#FFFFFF)',
                        inline: true,
                      },
                      {
                        name: 'Job ID',
                        value: jobId,
                        inline: true,
                      }
                    )
                    .setColor(0x0f172a);

                  const components: ActionRowBuilder<ButtonBuilder>[] = [];
                  if (uploadedDriveLink) {
                    components.push(
                      new ActionRowBuilder<ButtonBuilder>().addComponents(
                        new ButtonBuilder()
                          .setLabel('📂 فتح الملف على Google Drive')
                          .setStyle(ButtonStyle.Link)
                          .setURL(uploadedDriveLink)
                      )
                    );
                  }

                  await interaction.editReply({
                    embeds: [embed],
                    components,
                    files: filesToSend,
                  });
                } finally {
                  if (cleanupOutputPath && outputPath) {
                    fs.promises.unlink(outputPath).catch(() => {});
                  }
                }
                return;
              }

              const argParts = [
                '/clean',
                chapterOpt || 'chapter1',
                `sfx:${cleanSfx ? 'true' : 'false'}`,
              ];
              if (bubbleModeOpt) {
                argParts.push(`mode:${bubbleModeOpt}`);
              }
              const fullCmd = argParts.join(' ');
              const { botReply, activeChapterId } = await this.manager.executeBotCommand(fullCmd);

              const filesToSend: AttachmentBuilder[] = [];
              if (botReply.embed?.zipUrl) {
                const zipPath = this.manager.resolveDiskPathFromUrl(botReply.embed.zipUrl);
                if (fs.existsSync(zipPath) && fs.statSync(zipPath).size <= DISCORD_MAX_ATTACHMENT_BYTES) {
                  filesToSend.push(
                    new AttachmentBuilder(zipPath, {
                      name: `${activeChapterId || 'chapter'}_cleaned.zip`,
                    })
                  );
                }
              }

              await this.sendBotReplyEmbed(interaction, botReply, filesToSend);
              return;
            }

            const chapterOpt = interaction.options.getString('chapter') || '';
            const fullCmd = chapterOpt ? `${cmdName} ${chapterOpt}` : cmdName;
            const { botReply } = await this.manager.executeBotCommand(fullCmd);
            await this.sendBotReplyEmbed(interaction, botReply, []);
          } else if (interaction.isButton()) {
            const customId = interaction.customId;

            // --- /profile interactive buttons ---
            if (customId === 'profile:refresh') {
              const isAdmin = Boolean(
                interaction.memberPermissions?.has(PermissionFlagsBits.ManageGuild) ||
                  interaction.memberPermissions?.has(PermissionFlagsBits.Administrator)
              );
              const payload = this.buildProfilePayload(interaction.user.id, interaction.guildId, isAdmin);
              await interaction.update(payload);
              return;
            }

            if (customId === 'profile:link_user' || customId === 'profile:link_guild') {
              const scope = customId === 'profile:link_guild' ? 'guild' : 'user';
              const ownerId = scope === 'guild' ? interaction.guildId || interaction.user.id : interaction.user.id;
              const auth = createAuthUrl(scope, ownerId);
              if ('error' in auth) {
                await interaction.reply({ content: `❌ ${auth.error}`, ephemeral: true });
                return;
              }
              const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
                new ButtonBuilder()
                  .setLabel('🔗 تسجيل الدخول بحساب Google')
                  .setStyle(ButtonStyle.Link)
                  .setURL(auth.url)
              );
              const hint = auth.isLocalhostManual
                ? 'بعد ما توافق على الصلاحيات، المتصفح هيحولك على `http://localhost/?code=...` — انسخ الرابط من شريط العنوان ودوس زر **📋 إدخال كود الربط** في `/profile`.'
                : 'بعد ما توافق على الصلاحيات هيتم الربط تلقائياً، ارجع ودوس **🔄 تحديث** في `/profile`.';
              await interaction.reply({
                content: `🔐 **رابط ربط Google Drive (${scope === 'guild' ? 'للسيرفر' : 'لحسابك'})** (صالح 10 دقايق):\n${hint}`,
                components: [row],
                ephemeral: true,
              });
              return;
            }

            if (customId === 'profile:unlink_user' || customId === 'profile:unlink_guild') {
              const scope = customId === 'profile:unlink_guild' ? 'guild' : 'user';
              const ownerId = scope === 'guild' ? interaction.guildId || interaction.user.id : interaction.user.id;
              unlinkDrive(scope, ownerId);
              const isAdmin = Boolean(
                interaction.memberPermissions?.has(PermissionFlagsBits.ManageGuild) ||
                  interaction.memberPermissions?.has(PermissionFlagsBits.Administrator)
              );
              const payload = this.buildProfilePayload(interaction.user.id, interaction.guildId, isAdmin);
              await interaction.update(payload);
              return;
            }

            if (customId.startsWith('profile:paste_code:')) {
              const scope = customId.endsWith(':guild') ? 'guild' : 'user';
              const modal = new ModalBuilder()
                .setCustomId(`modal:drive_code:${scope}`)
                .setTitle('📋 إكمال ربط Google Drive');
              const input = new TextInputBuilder()
                .setCustomId('code_input')
                .setLabel('الصق رابط http://localhost/?code=... أو الكود')
                .setStyle(TextInputStyle.Short)
                .setRequired(true);
              modal.addComponents(new ActionRowBuilder<TextInputBuilder>().addComponents(input));
              await interaction.showModal(modal);
              return;
            }

            if (customId.startsWith('profile:set_folder:')) {
              const scope = customId.endsWith(':guild') ? 'guild' : 'user';
              const modal = new ModalBuilder()
                .setCustomId(`modal:drive_folder:${scope}`)
                .setTitle('📁 تحديد فولدر رفع Google Drive');
              const input = new TextInputBuilder()
                .setCustomId('folder_input')
                .setLabel('رابط فولدر Google Drive (أو الـ Folder ID)')
                .setStyle(TextInputStyle.Short)
                .setRequired(true);
              modal.addComponents(new ActionRowBuilder<TextInputBuilder>().addComponents(input));
              await interaction.showModal(modal);
              return;
            }

            if (customId.startsWith('cmd:')) {
              await interaction.deferReply();
              const cmd = customId.replace(/^cmd:/, '');
              const { botReply } = await this.manager.executeBotCommand(cmd);
              await this.sendBotReplyEmbed(interaction, botReply, []);
            }
          }
        } catch (err) {
          console.error('Discord interaction error:', err);
          if (interaction.isRepliable()) {
            const msg = `Cleaning error: ${(err as Error).message}`;
            if (interaction.deferred || interaction.replied) {
              await interaction.editReply({ content: msg }).catch(() => {});
            } else {
              await interaction.reply({ content: msg, ephemeral: true }).catch(() => {});
            }
          }
        }
      });

      await this.client.login(token);
    } catch (err) {
      console.warn('Discord bot login skipped or failed:', (err as Error).message);
      this.connected = false;
    }
  }

  private async sendBotReplyEmbed(
    interaction: { editReply: (payload: Record<string, unknown>) => Promise<unknown> },
    botReply: Awaited<ReturnType<ChapterCleaningManager['executeBotCommand']>>['botReply'],
    files: AttachmentBuilder[]
  ): Promise<void> {
    const embed = new EmbedBuilder()
      .setTitle(botReply.embed?.title || 'BearBot Scanlation Cleaner')
      .setDescription(botReply.text)
      .setColor(0x0f172a);

    if (botReply.embed?.fields) {
      for (const f of botReply.embed.fields) {
        embed.addFields({ name: f.label, value: f.value, inline: true });
      }
    }

    const components: ActionRowBuilder<ButtonBuilder>[] = [];
    if (botReply.embed?.actions && botReply.embed.actions.length > 0) {
      const row = new ActionRowBuilder<ButtonBuilder>();
      for (const act of botReply.embed.actions.slice(0, 4)) {
        row.addComponents(
          new ButtonBuilder()
            .setCustomId(`cmd:${act.command}`)
            .setLabel(act.label)
            .setStyle(act.variant === 'primary' ? ButtonStyle.Primary : ButtonStyle.Secondary)
        );
      }
      components.push(row);
    }

    await interaction.editReply({
      embeds: [embed],
      components,
      files,
    });
  }

  private async registerSlashCommands(token: string, clientId: string, guildIds: string[] = []): Promise<void> {
    const commands = [
      new SlashCommandBuilder()
        .setName('profile')
        .setDescription('ربط حساب Google Drive الخاص بك أو بالسيرفر للاستقبال والرفع المباشر'),
      new SlashCommandBuilder()
        .setName('clean')
        .setDescription('Clean an image, chapter ZIP, or Google Drive link with explicit SFX choice')
        .addBooleanOption((opt) =>
          opt
            .setName('clean_sfx')
            .setDescription('Clean Sound Effects (SFX)? True = clean SFX, False = keep SFX untouched')
            .setRequired(true)
        )
        .addAttachmentOption((opt) =>
          opt
            .setName('file')
            .setDescription('Optional PNG/JPG/WebP page or full-chapter ZIP archive')
            .setRequired(false)
        )
        .addStringOption((opt) =>
          opt
            .setName('drive_link')
            .setDescription('Google Drive link (folder of images, ZIP archive, or single image)')
            .setRequired(false)
        )
        .addBooleanOption((opt) =>
          opt
            .setName('upload_to_drive')
            .setDescription('Upload cleaned result to your linked Google Drive (auto when using drive_link)')
            .setRequired(false)
        )
        .addStringOption((opt) =>
          opt
            .setName('output_drive_folder')
            .setDescription('Optional target Google Drive folder link for uploading the cleaned result')
            .setRequired(false)
        )
        .addStringOption((opt) =>
          opt
            .setName('chapter')
            .setDescription('Mounted chapter ID (chapter1, chapter2, chapter3)')
            .setRequired(false)
        )
        .addStringOption((opt) =>
          opt
            .setName('bubble_mode')
            .setDescription('Speech bubble cleaning mode')
            .addChoices(
              { name: 'Smart Bubble Whitening (#FFFFFF)', value: 'whiten' },
              { name: 'Text-Only Removal', value: 'text_only' }
            )
            .setRequired(false)
        ),
      new SlashCommandBuilder()
        .setName('detect')
        .setDescription('Detect Speech Bubbles, OT, ST, and SFX regions without modifying pixels')
        .addStringOption((opt) =>
          opt.setName('chapter').setDescription('Chapter ID (chapter1, chapter2, chapter3)')
        ),
      new SlashCommandBuilder()
        .setName('review')
        .setDescription('Inspect detected regions and cleaning results')
        .addStringOption((opt) => opt.setName('chapter').setDescription('Chapter ID')),
      new SlashCommandBuilder()
        .setName('status')
        .setDescription('Show active chapter processing progress and worker status'),
      new SlashCommandBuilder()
        .setName('cancel')
        .setDescription('Cancel a running chapter cleaning task')
        .addStringOption((opt) => opt.setName('chapter').setDescription('Chapter ID')),
      new SlashCommandBuilder()
        .setName('settings')
        .setDescription('Configure cleaning categories, bubble mode, and engine routing')
        .addStringOption((opt) =>
          opt.setName('chapter').setDescription('Setting preset (mode:whiten, mode:text_only, toggle_sfx)')
        ),
      new SlashCommandBuilder()
        .setName('export')
        .setDescription('Package and export cleaned chapter pages as a lossless PNG ZIP archive')
        .addStringOption((opt) => opt.setName('chapter').setDescription('Chapter ID')),
      new SlashCommandBuilder()
        .setName('help')
        .setDescription('Display BearBot scanlation cleaning instructions and command reference'),
    ].map((c) => c.toJSON());

    const rest = new REST({ version: '10' }).setToken(token);
    await rest.put(Routes.applicationCommands(clientId), { body: commands });

    const targets = new Set<string>(guildIds);
    const envGuildId = (process.env.DISCORD_GUILD_ID || '').trim();
    if (envGuildId) targets.add(envGuildId);

    for (const gid of targets) {
      await rest.put(Routes.applicationGuildCommands(clientId, gid), { body: commands }).catch((err) => {
        console.warn(`Guild command sync skipped for ${gid}:`, (err as Error).message);
      });
    }
  }
}
