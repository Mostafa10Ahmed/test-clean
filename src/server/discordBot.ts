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
} from 'discord.js';
import { ChapterCleaningManager } from './chapterProcessor';
import { BubbleCleanMode } from '../engine/types';

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

  public async startIfConfigured(): Promise<void> {
    const token = process.env.DISCORD_BOT_TOKEN || process.env.DISCORD_TOKEN;
    if (!token || token.trim() === '') {
      return;
    }

    try {
      this.client = new Client({
        intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages],
      });

      this.client.once('ready', async (readyClient) => {
        this.connected = true;
        this.botTag = readyClient.user.tag;
        console.log(`Discord bot connected as ${this.botTag}`);
        await this.registerSlashCommands(token, readyClient.user.id);
      });

      this.client.on('interactionCreate', async (interaction) => {
        try {
          if (interaction.isChatInputCommand()) {
            await interaction.deferReply();
            const cmdName = `/${interaction.commandName}`;

            // Dedicated /clean handler with explicit clean_sfx boolean + optional file upload
            if (interaction.commandName === 'clean') {
              const cleanSfx = interaction.options.getBoolean('clean_sfx', true);
              const attachment = interaction.options.getAttachment('file');
              const chapterOpt = interaction.options.getString('chapter') || '';
              const bubbleModeOpt = interaction.options.getString('bubble_mode');
              const bubbleCleanMode: BubbleCleanMode | undefined =
                bubbleModeOpt === 'text_only'
                  ? 'TEXT_ONLY'
                  : bubbleModeOpt === 'whiten'
                    ? 'WHITEN'
                    : undefined;

              if (attachment) {
                const response = await fetch(attachment.url);
                if (!response.ok) {
                  throw new Error(`Failed to download attached file (${response.status})`);
                }
                const arrayBuf = await response.arrayBuffer();
                const fileName = attachment.name || 'page.png';
                const job = await this.manager.ingestUploadedFiles(
                  [{ originalname: fileName, buffer: Buffer.from(arrayBuf) }],
                  `Discord: ${fileName}`,
                  { enableSFX: cleanSfx, bubbleCleanMode }
                );

                const filesToSend: AttachmentBuilder[] = [];
                if (job.pages.length === 1 && job.pages[0].cleanedUrl) {
                  const pngPath = this.manager.resolveDiskPathFromUrl(job.pages[0].cleanedUrl);
                  if (fs.existsSync(pngPath) && fs.statSync(pngPath).size <= DISCORD_MAX_ATTACHMENT_BYTES) {
                    filesToSend.push(
                      new AttachmentBuilder(pngPath, {
                        name: `${job.pages[0].fileName.replace(/\.[^.]+$/, '')}_cleaned.png`,
                      })
                    );
                  }
                } else if (job.zipDownloadUrl) {
                  const zipPath = this.manager.resolveDiskPathFromUrl(job.zipDownloadUrl);
                  if (fs.existsSync(zipPath) && fs.statSync(zipPath).size <= DISCORD_MAX_ATTACHMENT_BYTES) {
                    filesToSend.push(
                      new AttachmentBuilder(zipPath, { name: `${job.id}_cleaned.zip` })
                    );
                  }
                }

                const embed = new EmbedBuilder()
                  .setTitle(`Cleaned: ${fileName}`)
                  .setDescription(
                    `Processed **${job.totalPages}** page(s) · **${job.cleanedRegions}** regions cleaned in **${(job.totalProcessingTimeMs / 1000).toFixed(2)}s**.`
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
                        job.settings.bubbleCleanMode === 'TEXT_ONLY'
                          ? 'Text-Only Removal'
                          : 'Smart Whitening (#FFFFFF)',
                      inline: true,
                    },
                    {
                      name: 'Job ID',
                      value: job.id,
                      inline: true,
                    }
                  )
                  .setColor(0x0f172a);

                await interaction.editReply({
                  embeds: [embed],
                  files: filesToSend,
                });
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
            .setStyle(
              act.variant === 'primary' ? ButtonStyle.Primary : ButtonStyle.Secondary
            )
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

  private async registerSlashCommands(token: string, clientId: string): Promise<void> {
    const commands = [
      new SlashCommandBuilder()
        .setName('clean')
        .setDescription('Clean an image, chapter ZIP, or mounted chapter with explicit SFX choice')
        .addBooleanOption((opt) =>
          opt
            .setName('clean_sfx')
            .setDescription('Clean Sound Effects (SFX)? True = clean SFX, False = keep SFX untouched')
            .setRequired(true)
        )
        .addAttachmentOption((opt) =>
          opt
            .setName('file')
            .setDescription('Optional PNG/JPG/WebP page or full-chapter ZIP archive to clean directly')
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
  }
}
